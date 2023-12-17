import os
import inspect
import pandas as pd
from statistics import mean, stdev

from CybORG import CybORG
from CybORG.Agents import BlueReactRemoveAgent, BlueReactRestoreAgent, BlueIntegrityRestoreAgent, BlueAvailabilityRestoreAgent, BlueIntegrityMisinformAgent
from CybORG.Agents import BlueAvailabilityMisinformAgent, BlueIntegrityMisinformAgent, BlueReactMisinformAgent
from CybORG.Shared.Scenarios.FileReaderScenarioGenerator import \
    FileReaderScenarioGenerator

MAX_STEPS_PER_GAME = 30
MAX_EPS = 334

def _create_env(map_name):
    # Get the directory containing cyborg
    cyborg_dir = os.path.dirname(os.path.dirname(inspect.getfile(CybORG)))
    path = cyborg_dir + f'/CybORG/Shared/Scenarios/scenario_files/{map_name}.yaml'
    norm_path = os.path.normpath(path)

    # Make scenario from specified file
    sg = FileReaderScenarioGenerator(norm_path)
    cyborg = CybORG(scenario_generator=sg, time_limit=MAX_STEPS_PER_GAME)
    return cyborg

if __name__ == "__main__":
    """
    This file runs heuristic agents made for vanilla CybORG using the true game state.
    """
    df_scores = pd.DataFrame()

    verbose = False

    scenarios = {
                'confidentiality_medium': 
                    {'Blue0' : BlueReactRestoreAgent(), 'Blue1' : BlueReactRestoreAgent()},
                'integrity_medium': 
                    {'Blue0' : BlueIntegrityRestoreAgent(), 'Blue1' : BlueIntegrityRestoreAgent()},
                'availability_medium': 
                    {'Blue0' : BlueAvailabilityRestoreAgent(), 'Blue1' : BlueAvailabilityRestoreAgent()},
    }
    scenarios = {
                'confidentiality_medium_misinform': 
                    {'Blue0' : BlueReactMisinformAgent(), 'Blue1' : BlueReactMisinformAgent()},
                'integrity_medium_misinform': 
                    {'Blue0' : BlueIntegrityMisinformAgent(), 'Blue1' : BlueIntegrityMisinformAgent()},
                'availability_medium_misinform': 
                    {'Blue0' : BlueAvailabilityMisinformAgent(), 'Blue1' : BlueAvailabilityMisinformAgent()},
    }
    #scenarios = {
    #            'integrity_medium': 
    ##                {'Blue0' : BlueIntegrityMisinformAgent(), 'Blue1' : BlueIntegrityMisinformAgent()},
    #}

    for scenario, blue_team in scenarios.items():
        env = _create_env(map_name=scenario)

        ave_reward =[]
        stat_rewards = []
        scores = []
        output = {}
        
        agents = list(blue_team.keys())
        agents.append('Red')

        # for each episode
        for ntrial in range(MAX_EPS): 
            score = 0
            output[ntrial] = {}
            
            # (re)set initial state for each agent
            obs = {agent: env.reset() for agent in blue_team.keys()}
            
            # for each turn in episode
            for turn in range(MAX_STEPS_PER_GAME):

                action_space = {agent: env.get_action_space(agent) for agent in blue_team}
                actions = {name: agent.get_action(obs[name], action_space[name]) for name, agent in blue_team.items()}
                    
                results = env.multi_step(actions=actions)
                obs = {agent: results[agent].observation for agent in blue_team.keys()}

                score += round(results[list(blue_team.keys())[0]].reward, 1)
            
                if verbose:
                    print(f"Turn: {turn}    Score: {score}")
            
            print(f'Episode score: {score}')
            scores.append(score)
            data = {'algorithm': 'heuristic', 'map_name': scenario, 'return_mean': score}
            df = pd.DataFrame(data, index=[0])
            df_scores = pd.concat([df_scores, df])
        average = sum(scores)/len(scores)
        print(f'Scenario: {scenario} Average score: {average} Std dev: {stdev(scores)}')
    df_scores.to_csv(f'./results/datasets/heuristic/misinform_heuristic.csv')

