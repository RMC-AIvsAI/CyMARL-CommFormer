import random
import pandas as pd
from statistics import mean, stdev

from CybORG.Agents import BlueReactRemoveAgent, BlueReactRestoreAgent, SleepAgent
# PyMARL interface with MA Wrapper
from pymarl2.envs.cyborg.cyborg_env import CyborgMultiAgentEnv

MAX_STEPS_PER_GAME = 30
MAX_EPS = 1000

if __name__ == "__main__":
    """
    This test file is used for debugging wrappers.
    It can run sleep and random agents.
    """
    
    sleep = False
    verbose = False
    df_scores = pd.DataFrame()

    scenarios = {
                #'confidentiality_small':
                #    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},
                #'availability_small': 
                #    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},
                #'integrity_small':
                #    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},
                'confidentiality_medium':
                    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},
                'availability_medium': 
                    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},
                'integrity_medium':
                    {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent()},

    }

    #scenarios = {
    #    'confidentiality_large':
    #        {'Blue0' : SleepAgent(), 'Blue1' : SleepAgent(), 'Blue2' : SleepAgent(), 'Blue3' : SleepAgent()},
    #}


    for scenario, blue_team in scenarios.items():
        env = CyborgMultiAgentEnv(map_name=scenario, episode_trace=False, time_limit=30, wrapper_type='table', no_obs=True)

        ave_reward =[]
        stat_rewards = []
        scores = []
        output = {}

        # for each episode
        for ntrial in range(MAX_EPS): 
            score = 0
            output[ntrial] = {}
            
            # (re)set initial state
            env.reset()

            # for each turn in episode
            for turn in range(MAX_STEPS_PER_GAME):
                # clear actions at each step to allow for default red & green behaviour
                action_space = env.get_avail_actions()
                possible_actions = [[] for agent in blue_team]
                for i, agent in enumerate(blue_team):
                    for j, action in enumerate(action_space[i]):
                        if action == 1:
                            possible_actions[i].append(j)

                if sleep:
                    actions = [0 for i, agent in enumerate(blue_team)]
                else:
                    actions = [random.choice(possible_actions[i]) for i, agent in enumerate(blue_team)]

                # Results are unlabelled and only include relevant info for pymarl
                results = env.step(actions=actions)

                score += results[0]

                if verbose:
                    print(f"Turn: {turn}    Score: {score}")
            
            print(f'Episode score: {score}')
            scores.append(score)
            data = {'algorithm': 'random', 'map_name': scenario, 'return_mean': score}
            df = pd.DataFrame(data, index=[0])
            df_scores = pd.concat([df_scores, df])
        average = sum(scores)/len(scores)
        print(f'Scenario: {scenario} Average score: {average} Std dev: {stdev(scores)}')
    df_scores.to_csv(f'./results/datasets/heuristic/all_random.csv')


