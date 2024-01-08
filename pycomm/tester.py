from envs.cyborg import CyborgEnv
import torch
import numpy as np

env = CyborgEnv(map_name="confidentiality_small", wrapper_type="vector")

state = env.reset()





combined_actions = torch.zeros(env.n_agents, dtype=torch.long)
for n in range(30):

    action_range_1, comm_range_1 = env.get_action_range(10, 0, 0)
    action_range_2, comm_range_2 = env.get_action_range(10, 0, 1)
    a_range_1 = range(action_range_1[0].item() - 1, action_range_1[1].item())
    a_range_2 = range(action_range_2[0].item() - 1, action_range_2[1].item())
    #action_agent_1 = 1
    #action_agent_2 = 1
    #action_agent_1 = torch.from_numpy(np.random.choice(a_range_1, 1)).item()
    #action_agent_2 = torch.from_numpy(np.random.choice(a_range_2, 1)).item()
    print(action_range_1, action_range_2)
    action_agent_1 = int(input("Enter action for Agent 1: "))  # Take user input for Agent 1
    action_agent_2 = int(input("Enter action for Agent 2: "))  # Take user input for Agent 2

    combined_actions[0] = action_agent_1
    combined_actions[1] = action_agent_2

    reward, terminal = env.step(combined_actions)
    state = env.get_state()

    print("Reward:", reward)

    print("State:", state)
