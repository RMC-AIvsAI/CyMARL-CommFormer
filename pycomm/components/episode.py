import torch
import copy
from utils.dotdic import DotDic

class Episode:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.episodes = []

    def reset(self):
        self.episodes = []

    def add_episode(self, episode_batch):
        self.episodes.append(episode_batch)
    
    def combine_episodes(self):
        if not self.episodes:
            return None
        combined_batch = DotDic({})
        steps_tensors, ended_tensors, r_tensors = [], [], []

        for episode_batch in self.episodes:
            steps_tensors.append(episode_batch.steps)
            ended_tensors.append(episode_batch.ended)
            r_tensors.append(episode_batch.r)
        combined_batch.steps = torch.cat(steps_tensors, dim=0)
        combined_batch.ended = torch.cat(ended_tensors, dim=0)
        combined_batch.r = torch.cat(r_tensors, dim=0)

        combined_batch.step_records = []
        for step in range(self.opt.nsteps):
            combined_batch.step_records.append(DotDic({}))
            state_tensors, r_t_tensors, terminal_tensors, a_t_tensors = [], [], [], []
            comm_tensors, comm_target_tensors, q_a_t_tensors, q_a_max_t_tensors = [], [], [],[]
            for episode_batch in self.episodes:
                state_tensors.append(episode_batch.step_records[step].s_t)
                r_t_tensors.append(episode_batch.step_records[step].r_t)
                terminal_tensors.append(episode_batch.step_records[step].terminal)
                a_t_tensors.append(episode_batch.step_records[step].a_t)
                if self.opt.comm_enabled:
                    comm_tensors.append(episode_batch.step_records[step].comm)
                    comm_target_tensors.append(episode_batch.step_records[step].comm_target)
                q_a_t_tensors.append(episode_batch.step_records[step].q_a_t)
                q_a_max_t_tensors.append(episode_batch.step_records[step].q_a_max_t)
            combined_batch.step_records[step].s_t = torch.cat(state_tensors, dim=0)
            combined_batch.step_records[step].r_t = torch.cat(r_t_tensors, dim=0)
            combined_batch.step_records[step].terminal = torch.cat(terminal_tensors, dim=0)
            combined_batch.step_records[step].a_t = torch.cat(a_t_tensors, dim=0)
            if self.opt.comm_enabled:
                combined_batch.step_records[step].comm = torch.cat(comm_tensors, dim=0)
                combined_batch.step_records[step].comm_target = torch.cat(comm_target_tensors, dim=0)
            combined_batch.step_records[step].q_a_t = torch.cat(q_a_t_tensors, dim=0)
            combined_batch.step_records[step].q_a_max_t = torch.cat(q_a_max_t_tensors, dim=0)
        
        return combined_batch
    
    def create_episode(self, batch_size):
        opt = self.opt
        episode = DotDic({})
        episode.steps = torch.zeros(batch_size).int().to(self.device)
        episode.ended = torch.zeros(batch_size).int().to(self.device)
        episode.r = torch.zeros(batch_size, opt.game_nagents).float().to(self.device)
        episode.step_records = []
        return episode

    def create_step_record(self, batch_size):
        opt = self.opt
        record = DotDic({})
        record.s_t = None
        record.r_t = torch.zeros(batch_size, opt.game_nagents).to(self.device)
        record.terminal = torch.zeros(batch_size).to(self.device)

        record.agent_inputs = []

        # Track actions at time t per agent
        record.a_t = torch.zeros(batch_size, opt.game_nagents, dtype=torch.long).to(self.device)

        # Track messages sent at time t per agent
        if opt.comm_enabled:
            comm_dtype = opt.model_dial and torch.float or torch.long
            comm_dtype = torch.float
            record.comm = torch.zeros(batch_size, opt.game_nagents, opt.game_comm_bits, dtype=comm_dtype).to(self.device)
            if opt.model_dial and opt.model_target:
                record.comm_target = record.comm.clone()

        # Track hidden state per time t per agent
        record.hidden = torch.zeros(opt.game_nagents, opt.model_rnn_layers, batch_size, opt.model_rnn_size).to(self.device)
        record.hidden_target = torch.zeros(opt.game_nagents, opt.model_rnn_layers, batch_size, opt.model_rnn_size).to(self.device)

        # Track Q(a_t) and Q(a_max_t) per agent
        record.q_a_t = torch.zeros(batch_size, opt.game_nagents).to(self.device)
        record.q_a_max_t = torch.zeros(batch_size, opt.game_nagents).to(self.device)

        return record
    
class PlayGame:
    def __init__(self, opt, episode, filename):
        self.opt = opt
        self.episode = episode
        self.filename = filename
        self.file = None
        self._setup()

    def open_file(self):
        self.file = open(self.filename, 'w')

    def close_file(self):
        if self.file is not None:
            self.file.close()

    def _setup(self):
        self.action_dict_1 = {}
        self.action_dict_1[1] = "Monitor"
        self.action_dict_1[2] = "Remove User1"
        self.action_dict_1[3] = "Remove User2"
        self.action_dict_1[4] = "Restore User1"
        self.action_dict_1[5] = "Restore User2"
        self.action_dict_1[6] = "Analyse User1"
        self.action_dict_1[7] = "Analyse User2"

        self.action_dict_2 = {}
        self.action_dict_2[1] = "Monitor"
        self.action_dict_2[2] = "Remove Op_Host0"
        self.action_dict_2[3] = "Remove Op_Server0"
        self.action_dict_2[4] = "Restore Op_Host0"
        self.action_dict_2[5] = "Restore Op_Server0"
        self.action_dict_2[6] = "Analyse Op_Host0"
        self.action_dict_2[7] = "Analyse Op_Server0"

    def play_game(self):
        if self.file is not None:
            for i in range(8):
                previous_step_record = None
                for j, step_record in enumerate(self.episode.step_records):
                    self.file.write(f"---- Turn {j} ----\n")
                    if j != 0:
                        if self.opt.comm_enabled:
                            message = previous_step_record.comm[i].tolist()
                        
                            self.file.write(f"Message sent: {message}\n")

                        a = previous_step_record.a_t[i].tolist()
                        actions = [self.action_dict_1[a[0]], self.action_dict_2[a[1]]]
                        self.file.write(f"Action: {actions}\n")

                        reward = previous_step_record.r_t[i].tolist()
                        self.file.write(f"Reward: {reward}\n")

                    if step_record.s_t is not None:
                        state = step_record.s_t[i].tolist()
                        self.file.write(f"State: {state}\n")
                    self.file.write("\n")
                    previous_step_record = step_record

                self.file.write("\n")
                total_reward = self.episode.r[i].sum().item()/2
                self.file.write(f"Total Reward: {total_reward}\n")
                self.file.write("\n")          
                    