import time
import numbers
import numpy as np
import torch
import os
from commformer.runner.shared.base_runner import Runner

def _t2n(x):
    """
    Converts a PyTorch tensor to a NumPy array.

    This function detaches the tensor from the current computational graph,
    moves it to the CPU if it's not already there, and converts it to a NumPy array.

    Args:
        x (torch.Tensor): The input tensor to be converted.

    Returns:
        numpy.ndarray: The converted NumPy array.
    """
    return x.detach().cpu().numpy()

def merge_stat(src, dest):
    """
    Merges statistics from the source dictionary into the destination dictionary.

    Parameters:
    src (dict): The source dictionary containing statistics to merge.
    dest (dict): The destination dictionary where statistics will be merged into.

    The function handles different types of values:
    - If the key does not exist in the destination dictionary, it is added.
    - If the value is a number, it is added to the existing value in the destination dictionary.
    - If the value is a numpy ndarray, it is added to the existing value in the destination dictionary.
    - If the value is a list, it is extended or appended to the existing list in the destination dictionary.
    - For other types, the values are combined into a list.
    """
    for k, v in src.items():
        if not k in dest:
            dest[k] = v
        elif isinstance(v, numbers.Number):
            dest[k] = dest.get(k, 0) + v
        elif isinstance(v, np.ndarray): # for rewards in case of multi-agent
            dest[k] = dest.get(k, 0) + v
        else:
            if isinstance(dest[k], list) and isinstance(v, list):
                dest[k].extend(v)
            elif isinstance(dest[k], list):
                dest[k].append(v)
            else:
                dest[k] = [dest[k], v]

class CybORGRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(CybORGRunner, self).__init__(config)

    def run(self):
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            # reset environment and buffer
            # CybORG requires a full reset each episode
            self.warmup()
            episode_start_time = time.time()
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            step_times = []  # List to store time taken for each step

            # save comms channels matrix for the episode, used by save_actions_to_file function
            comms_channels = _t2n(self.trainer.policy.transformer.edge_return(exact=True))
            # add self loop to the comms channels matrix if required
            if self.all_args.self_loop_add:
                comms_channels = comms_channels + np.eye(comms_channels.shape[0])
                comms_channels = np.clip(comms_channels, 0, 1)

            for step in range(self.episode_length):
                step_start_time = time.time()
                
                # Sample actions
                # rnn_states and rnn_states_critic are only included for compatibility purposes, they will always be zeros
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Observe reward, next obs and available actions
                obs, rewards, dones, infos = self.envs.step(actions_env)
                available_actions = np.array(self.envs.get_avail_actions(step), dtype=int)

                # CybORG specific dimension adjustment
                rewards = np.expand_dims(rewards, -1)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions

                # insert data into buffer
                self.insert(data)

                step_end_time = time.time()
                step_times.append(step_end_time - step_start_time)  # Store the time taken for this step

            # Compute the average time per step
            avg_step_time = sum(step_times) / len(step_times)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                time_elapsed_h = int((end - start) // 3600)
                time_elapsed_m = int((end - start) % 3600 // 60)
                estimated_time_seconds = (end - episode_start_time) * (episodes - episode)
                estimated_time_h = int(estimated_time_seconds // 3600)
                estimated_time_m = int((estimated_time_seconds % 3600) // 60)
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n \
                      Time Elapsed: {}h{}m. Estimated time to complete: {}h{}m. Average step time: {}\n"
                        .format(
                                self.all_args.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start)),
                                time_elapsed_h,
                                time_elapsed_m,
                                estimated_time_h,
                                estimated_time_m,
                                avg_step_time))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}\n".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)

                edges = _t2n(self.trainer.policy.transformer.edges)
                print(edges)
                edges = _t2n(self.trainer.policy.transformer.edge_return(exact=True))
                image = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0)
                self.writter.add_image('Matrix', image, dataformats='NCHW', global_step=total_num_steps)

                # Save actions to file
                self.save_actions_to_file(os.path.join(self.train_act_dir, "actions_output_ep_" + str(episode) + ".txt"), comms_channels)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, episode)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        available_actions = np.array(self.envs.get_avail_actions(0), dtype=int)

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete': # not used by CybORG
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = actions.reshape(self.n_rollout_threads, self.num_agents) + 1 # add 1 to match the CybORG action space

        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        infos = list(infos)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, infos, available_actions=available_actions)

    @torch.no_grad()
    def eval(self, total_num_steps, last_training_episode):
        # reset eval env and determine available actions
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()
        eval_available_actions = np.array(self.eval_envs.get_avail_actions(0), dtype=int)
        
        # initialize eval buffers, used to output actions to file
        eval_buffer_obs = np.zeros((self.all_args.eval_episode_length + 1, self.n_eval_rollout_threads, self.num_agents, self.buffer.obs.shape[3]), dtype=np.float32)
        eval_buffer_obs[0] = eval_obs.copy()
        eval_buffer_rew = np.zeros((self.all_args.eval_episode_length, self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_buffer_act = np.zeros((self.all_args.eval_episode_length, self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_buffer_infos = np.empty((self.all_args.eval_episode_length, self.n_eval_rollout_threads), dtype=object)

        # share_obs will remain 0 throughout the eval process as the eval env is not used for training
        if self.use_centralized_V:
            eval_share_obs = eval_obs.reshape(self.n_eval_rollout_threads, -1)
            eval_share_obs = np.expand_dims(eval_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            eval_share_obs = eval_obs

        # rnn states and masks are only included for compatibility purposes, they will always be zeros and ones respectively
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_steps = []
        one_episode_steps = [0 for _ in range(self.all_args.eval_episodes)]
        flag = [False for _ in range(self.all_args.eval_episodes)]

        eval_comms_channels = _t2n(self.trainer.policy.transformer.edge_return(exact=True))
        if self.all_args.self_loop_add:
            eval_comms_channels = eval_comms_channels + np.eye(eval_comms_channels.shape[0])
            eval_comms_channels = np.clip(eval_comms_channels, 0, 1)

        for eval_step in range(self.all_args.eval_episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                          np.concatenate(eval_share_obs),
                          np.concatenate(eval_obs),
                          np.concatenate(eval_rnn_states),
                          np.concatenate(eval_masks),
                          np.concatenate(eval_available_actions),
                          deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete': # not used by CybORG
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = eval_actions.reshape(self.n_eval_rollout_threads, self.num_agents) + 1 # add 1 to match the CybORG action space
            else:
                raise NotImplementedError

            # Observe reward, next obs and available actions
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_available_actions = np.array(self.eval_envs.get_avail_actions(eval_step), dtype=int)

            # CybORG specific dimension adjustment
            eval_rewards_CybORG = np.expand_dims(eval_rewards, -1)

            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            one_episode_steps += np.array([1 for _ in range(self.all_args.eval_episodes)])

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones[eval_i][0] == True:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

                    eval_episode_steps.append(one_episode_steps[eval_i])
                    one_episode_steps[eval_i] = 0

                    flag[eval_i] = True
            
            eval_buffer_obs[eval_step + 1] = eval_obs.copy()
            eval_buffer_rew[eval_step] = eval_rewards_CybORG.copy()
            eval_buffer_act[eval_step] = eval_actions.copy()
            eval_buffer_infos[eval_step] = eval_infos

            if eval_episode >= self.all_args.eval_episodes:
                break


        if len(eval_episode_rewards) < self.all_args.eval_episodes:
            for eval_i in range(self.n_eval_rollout_threads):
                if flag[eval_i] == False:
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    eval_episode_steps.append(one_episode_steps[eval_i])
                
                if len(eval_episode_rewards) >= self.all_args.eval_episodes:
                    break
        
        key_average = '/eval_average_episode_rewards'
        key_max = '/eval_max_episode_rewards'
        key_steps = '/eval_average_steps'
        eval_env_infos = {key_average: eval_episode_rewards,
                            key_max: [np.max(eval_episode_rewards)],
                            key_steps: eval_episode_steps}
        self.log_env(eval_env_infos, total_num_steps)

        print("eval average episode rewards: {}, steps: {}"
                .format(np.mean(eval_episode_rewards), np.mean(eval_episode_steps)))
        
        # Save actions to file
        self.save_eval_actions_to_file(os.path.join(self.eval_act_dir, "actions_output_eval_ep_" + str(last_training_episode) + ".txt"), eval_comms_channels, eval_buffer_obs, eval_buffer_rew, eval_buffer_act, eval_buffer_infos)

    def save_actions_to_file(self, actions_file_path, comms_channels):
        """
        Reads the self.buffer.actions, rewards, obs and infos arrays and writes their contents in human readable format to a .txt file in the run_dir results directory.
        Active comms channels for the episode are also written to the file.
        """
         # Get the agent IDs from the environment
        agent_ids = self.envs.get_agent_ids()[0]
                
        # Iterate through all agent IDs and create action dictionaries dynamically
        self.action_dicts = {}
        for numeric_index, agent_id in enumerate(agent_ids):  # Iterate through all agent IDs with numeric indices
            self.action_dicts[numeric_index] = {index: value for index, value in enumerate(self.envs.get_possible_actions(agent_id)[0])}

        # Write contents to the file
        with open(actions_file_path, "w") as file:
            # Write active communication channels for the episode
            file.write(f"Active Communication Channels for the Episode:\n")
            
            # Define a fixed width for each column
            column_width = 10
            
            # Write column headers (agent IDs) with proper alignment
            file.write(" " * column_width + "".join(f"{agent_id:<{column_width}}" for agent_id in agent_ids) + "\n")
            
            # Write each row with proper alignment
            for row_label, row in zip(agent_ids, comms_channels):
                row_string = "".join(f"{int(value):<{column_width}}" for value in row)  # Format each value with fixed width
                file.write(f"{row_label:<{column_width}}{row_string}\n")  # Align the row label
            file.write("\n")

            # Iterate through threads first
            for thread_id in range(self.n_rollout_threads):
                file.write(f"Thread {thread_id}:\n")

                # Iterate through steps for the current thread
                for step in range(self.episode_length):
                    file.write(f"  Step {step}:\n")

                    # Iterate through agents for the current step and thread
                    for agent_id, thread_action in enumerate(self.buffer.actions[step][thread_id]):
                        action_string = self.action_dicts[agent_id].get(int(thread_action), "Unknown Action")

                        # Get the reward and observation for the current agent
                        reward = self.buffer.rewards[step][thread_id][agent_id]
                        obs = self.buffer.obs[step][thread_id][agent_id]

                        # Write the agent's action, reward, and observation to the file
                        file.write(f"    {agent_ids[agent_id]}: {action_string}, Reward: {reward}, Obs: {obs}\n")

                    # Get the Red Agent action for the current step and thread
                    # Red actions are taken after blue actions
                    red_action_string = self.buffer.infos[step][thread_id]
                    file.write(f"    Red Agent Action: {red_action_string}\n")
                    file.write("\n")

                # Sum rewards along the first axis (steps)
                rewards_per_thread = np.sum(self.buffer.rewards, axis=0) # sum over agents not required as agents receive team rewards

                # Print the total reward per thread at the end of the episode
                file.write(f"Thread {thread_id}: Total Reward = {rewards_per_thread[thread_id, 0]}\n")

                file.write("\n")

        print(f"\nTraining actions have been saved to {actions_file_path}\n")

    def save_eval_actions_to_file(self, actions_file_path, comms_channels, eval_buffer_obs, eval_buffer_rew, eval_buffer_act, eval_buffer_infos):
         # Get the agent IDs from the environment
        agent_ids = self.eval_envs.get_agent_ids()[0]
                
        # Iterate through all agent IDs and create action dictionaries dynamically
        self.action_dicts = {}
        for numeric_index, agent_id in enumerate(agent_ids):  # Iterate through all agent IDs with numeric indices
            self.action_dicts[numeric_index] = {index: value for index, value in enumerate(self.eval_envs.get_possible_actions(agent_id)[0])}

        # Write contents to the file
        with open(actions_file_path, "w") as file:
            # Write active communication channels for the episode
            file.write(f"Active Communication Channels for the Episode:\n")
            
            # Define a fixed width for each column
            column_width = 10
            
            # Write column headers (agent IDs) with proper alignment
            file.write(" " * column_width + "".join(f"{agent_id:<{column_width}}" for agent_id in agent_ids) + "\n")
            
            # Write each row with proper alignment
            for row_label, row in zip(agent_ids, comms_channels):
                row_string = "".join(f"{int(value):<{column_width}}" for value in row)  # Format each value with fixed width
                file.write(f"{row_label:<{column_width}}{row_string}\n")  # Align the row label
            file.write("\n")

            # Iterate through threads first
            for thread_id in range(self.n_eval_rollout_threads):
                file.write(f"Thread {thread_id}:\n")

                # Iterate through steps for the current thread
                for step in range(self.all_args.eval_episode_length):
                    file.write(f"  Step {step}:\n")

                    # Iterate through agents for the current step and thread
                    for agent_id, thread_action in enumerate(eval_buffer_act[step][thread_id]):
                        action_string = self.action_dicts[agent_id].get(int(thread_action), "Unknown Action")

                        # Get the reward and observation for the current agent
                        reward = eval_buffer_rew[step][thread_id][agent_id]
                        obs = eval_buffer_obs[step][thread_id][agent_id]

                        # Write the agent's action, reward, and observation to the file
                        file.write(f"    {agent_ids[agent_id]}: {action_string}, Reward: {reward}, Obs: {obs}\n")

                    # Get the Red Agent action for the current step and thread
                    # Red actions are taken after blue actions
                    red_action_string = eval_buffer_infos[step][thread_id]
                    file.write(f"    Red Agent Action: {red_action_string}\n")
                    file.write("\n")

                # Sum rewards along the first axis (steps)
                rewards_per_thread = np.sum(eval_buffer_rew, axis=0) # sum over agents not required as agents receive team rewards

                # Print the total reward per thread at the end of the episode
                file.write(f"Thread {thread_id}: Total Reward = {rewards_per_thread[thread_id, 0]}\n")

                file.write("\n")

        print(f"\nEvaluation actions have been saved to {actions_file_path}\n")
