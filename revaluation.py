import numpy as np
import torch

import os
from collections import namedtuple
import copy
import random
import json

import utils
from agents import *
from env import SimpleEnv



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

observation = namedtuple('observation', ('terminal', 'state', 'action', 'reward', 'next_state'))


HEIGHT = 11
WIDTH = 11
NUM_CHANNELS = 1
START_LOC = (4, 5)
BOUNDARIES = [[(1, 4), (9, 6)], [(7, 1), (9, 9)]]
REWARD_LOC = [(8, 2), (8, 8)]
REWARD_VALUES = [1, 10]

POSSIBLE_ACTIONS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
NUM_ACTIONS = len(POSSIBLE_ACTIONS)
CONTINOUS_TASK = False
SIMPLIFIED = True


#DEFINE TIME PARAMS
NUM_EXPLORATION_STEPS = int(2e4)
NUM_SIMULATION_STEPS = 20 
NUM_TASK_STEPS = int(1e3)
SIMULATION_UPDATE = 500
NUM_SIMULATIONS = 1
EPISODE_LENGTH = 200
CONVERGENCE_CRITERIA = 7
convergence_step = 0


#DEFINE THE HYPER PARAMETERS
BATCH_SIZE = 32
MEMORY_SIZE = int(1e3)
TARGET_UPDATE = 64    #Steps till target network gets updated
GAMMA = 0.99
SR_DIMENSIONS = 121


#DEFINE THE EXPLORATION STRATEGY
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 10000


#DEFINE THE OPTIMIZER PARAMETERS
LEARNING_RATE = 1e-3
EPSILON = 0.000009765625
WEIGHT_DECAY = 0.05


#DEFINE THE LOSS WEIGHTS
REWARD_WEIGHT = 9e-05
DECODER_WEIGHT = 0.01
SR_WEIGHT = 1-REWARD_WEIGHT


SEEDS = random.sample(range(1000), 10)

states_to_encode = []
for box in BOUNDARIES:
    for dim1 in range(box[0][0], box[1][0]+1):
        for dim2 in range(box[0][1], box[1][1]+1):
            states_to_encode.append((dim1, dim2))

if not os.path.isdir("Revaluation_simplified/"):
    os.mkdir("Revaluation_simplified/")



for SR_BIAS in np.arange(0, 1.01, 0.05):


    for START_LOC in [(2, 5), (7, 5)]: 
    
        print(f"STARTING EXPERIMENT WITH SR_BIAS: {SR_BIAS} AND STARTING STATE: {START_LOC}")

        if START_LOC==(2, 5):
            condition="distal/"
        elif START_LOC==(7,5):
            condition="close/"

        if not os.path.isdir("Revaluation_simplified/"+condition):
            os.mkdir("Revaluation_simplified/"+condition)       


        if not os.path.isdir("Revaluation_simplified/"+condition + str(SR_BIAS)):
            os.mkdir("Revaluation_simplified/"+condition + str(SR_BIAS))


        solved_revaluation_one_shot = 0

        for seed in SEEDS:
            torch.manual_seed(seed)
            random.seed(seed)


            ####Set up the objects####
            ##Set up the environment
            env = SimpleEnv(HEIGHT, WIDTH, BOUNDARIES, REWARD_LOC, REWARD_VALUES, POSSIBLE_ACTIONS, CONTINOUS_TASK, SIMPLIFIED)

            #Set up the agent
            agent_info = {
                "device": device,
                "channel": NUM_CHANNELS,
                "height": HEIGHT,
                "width": WIDTH,
                "batch_size": BATCH_SIZE,
                "num_actions": NUM_ACTIONS,
                "sr_dimensions": SR_DIMENSIONS,
                "gamma": GAMMA,
                "lr": LEARNING_RATE,
                "eps": EPSILON,
                "weight_decay": WEIGHT_DECAY,
                "strategy": utils.EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY),
                "sr_bias": SR_BIAS,
                "reward_weight": REWARD_WEIGHT,
                "decoder_weight": DECODER_WEIGHT,
                "sr_weight": SR_WEIGHT
                }


            agent = DSR_agent_no_encoder(agent_info)


            ##Create the memory
            observation = namedtuple('observation', ('terminal', 'state', 'action', 'reward', 'next_state'))
            memory = utils.ReplayMemory(MEMORY_SIZE)

            ##Set up tracking metrics
            losses = []
            reward_losses = []
            sr_losses = []
            u_losses = []
            kl_losses = []
            decoder_losses = []
            heatmaps = []
            rewards_per_episode = []
            state_embedding_map = []
            reward_map = []
            state_embeddings_over_time = []
            current_steps = 0
            states_visited = []

            #Exploration
            env.change_reward(reward_loc=REWARD_LOC, reward_value=[0, 0])
            env.move_agent_to_loc(random.choice(states_to_encode))
            env.change_continous(True)
            rewards_over_time = []
            current_steps = 0

            for step in range(1, 20000):

                current_steps += 1

                #Move the agent
                state = env.create_state().to(device)

                action_idx = torch.randint(0, NUM_ACTIONS, (1,))

                #Observe the interaction
                terminal, state, action, reward, next_state = env.create_observation(action_idx)

                #Move tensors to GPU
                terminal = terminal.to(device); state = state.to(device); action_idx = action_idx.reshape(-1).to(device); reward = reward.to(device); next_state = next_state.to(device)

                #Save observation to the replay memory
                if torch.any(state != next_state):
                    memory.push(observation(terminal, state, action_idx, reward, next_state))

                if current_steps%1000==0:
                    env.move_agent_to_loc(random.choice(states_to_encode))
                    current_steps = 0
                
                #Train the agent every 2 steps
                if (step>MEMORY_SIZE//10) & (step%2==0):
                    #Sample interactions from the replay memory
                    observations = memory.sample(BATCH_SIZE)
                    (loss, reward_loss, u_loss) = agent.train(observations)


                #Save the loss every 100 steps
                if (step%100==0) & (step>MEMORY_SIZE//10):
                    losses.append(loss)

                #Periodically update the weights of the target network
                if step%TARGET_UPDATE==0:
                    agent.update_target_network()

                

            ###Acquisition TASK
            env.change_reward(reward_loc=REWARD_LOC, reward_value=[1, 10])
            env.change_continous(False)
            rewards_over_time = []
            agent.set_w([(8*WIDTH + 2), (8*WIDTH + 8)], [1, 10])

            #Confront the agent with the change in reward
            for loc in [(7, 2), (8, 1), (9, 2), (8, 3), (7, 8), (8, 7), (9, 8), (8, 9)]:
                
                for repeat in range(10):
                    env.move_agent_to_loc(loc)
                    for step in range(100):
                        state = env.create_state().to(device)
                        action_idx = torch.randint(0, NUM_ACTIONS, (1,))

                        #Observe the interaction
                        terminal, state, action, reward, next_state = env.create_observation(action_idx)

                        #Move tensors to GPU
                        terminal = terminal.to(device); state = state.to(device); action_idx = action_idx.reshape(-1).to(device); reward = reward.to(device); next_state = next_state.to(device)

                        #Save observation to the replay memory
                        if torch.any(state != next_state):
                            memory.push(observation(terminal, state, action_idx, reward, next_state))


                        if terminal:
                            memory.push(observation(terminal, state, action_idx, reward, next_state))
                            break


            for train in range(20):
                observations = memory.sample(BATCH_SIZE)
                (loss, reward_loss, u_loss) = agent.train(observations)


            current_steps = 0
            agent.reset_epsilon()
            env.move_agent_to_loc(START_LOC)
            right_choice = 0

            for step in range(50000):

                current_steps += 1

                #Move the agent
                state = env.create_state().to(device)
                action_idx = agent.select_action_epsilon_greedy(state)

                #Observe the interaction
                terminal, state, action, reward, next_state = env.create_observation(action_idx)

                #Move tensors to GPU
                terminal = terminal.to(device); state = state.to(device); action_idx = action_idx.reshape(-1).to(device); reward = reward.to(device); next_state = next_state.to(device)

                #Save observation to the replay memory
                if torch.any(state != next_state):
                    memory.push(observation(terminal, state, action_idx, reward, next_state))

                if terminal:
                    memory.push(observation(terminal, state, action_idx, reward, next_state))
                    env.move_agent_to_loc(START_LOC)
                    current_steps = 0

                if current_steps%200==0:
                    #Reset condition
                    env.move_agent_to_loc(START_LOC)
                    current_steps = 0
                
                #Train the agent every 2 steps
                if (step>MEMORY_SIZE//10) & (step%2==0):
                    #Sample interactions from the replay memory
                    observations = memory.sample(BATCH_SIZE)
                    (loss, reward_loss, u_loss) = agent.train(observations)


                #Save the loss every 100 steps
                if (step%100==0) & (step>MEMORY_SIZE//10):
                    losses.append(loss)

                #Periodically update the weights of the target network
                if step%TARGET_UPDATE==0:
                    agent.update_target_network()


                #Periodically run a simulation to track the progress
                if (step%100==0):
                    
                    env_sim = copy.copy(env)
                    reward_count = 0

                    for simulation in range(NUM_SIMULATIONS):
                        env_sim.move_agent_to_loc(START_LOC)
                        state = env_sim.create_state().to(device)

                        for _ in range(NUM_SIMULATION_STEPS):

                            action_idx = agent.select_action_greedy(state)
                            terminal, state, action, reward_, next_state = env_sim.create_observation(action_idx)
                            reward_count += reward_.item()
                            states_visited.append(state.cpu().detach())
                            state = next_state.to(device)

                            if terminal:
                                break

                        if reward_count==10:
                            right_choice+=1
                        
                if right_choice==5:
                    break



            ###Revaluation TASK
            env.change_reward(reward_loc=REWARD_LOC, reward_value=[10, 1])
            agent.set_w([(8*WIDTH + 2), (8*WIDTH + 8)], [10, 1])
            env.change_continous(False)
            rewards_over_time = []

            #Confront the agent with the change in reward
            for loc in [(7, 2), (8, 1), (9, 2), (8, 3), (7, 8), (8, 7), (9, 8), (8, 9)]:
                
                for repeat in range(10):
                    env.move_agent_to_loc(loc)
                    for step in range(100):
                        state = env.create_state().to(device)
                        action_idx = torch.randint(0, NUM_ACTIONS, (1,))

                        #Observe the interaction
                        terminal, state, action, reward, next_state = env.create_observation(action_idx)

                        #Move tensors to GPU
                        terminal = terminal.to(device); state = state.to(device); action_idx = action_idx.reshape(-1).to(device); reward = reward.to(device); next_state = next_state.to(device)

                        #Save observation to the replay memory
                        if torch.any(state != next_state):
                            memory.push(observation(terminal, state, action_idx, reward, next_state))

                        if terminal:
                            memory.push(observation(terminal, state, action_idx, reward, next_state))
                            break


            for train in range(20):
                observations = memory.sample(BATCH_SIZE)
                (loss, reward_loss, u_loss) = agent.train(observations)

            current_step = 0
            agent.reset_epsilon()
            env.move_agent_to_loc(START_LOC)

            env_sim = copy.copy(env)
            reward_count = 0
            states_visited = []

            for simulation in range(NUM_SIMULATIONS):
                env_sim.move_agent_to_loc(START_LOC)
                state = env_sim.create_state().to(device)

                for _ in range(NUM_SIMULATION_STEPS):

                    action_idx = agent.select_action_greedy(state)
                    terminal, state, action, reward_, next_state = env_sim.create_observation(action_idx)
                    reward_count += reward_.item()
                    state = next_state.to(device)

                    if terminal:
                        break

            #Track the result of the simulation
            rewards_per_episode.append(reward_count/NUM_SIMULATIONS)


            if rewards_per_episode[-1]==10:
              solved_revaluation_one_shot += 1

        with open("Revaluation_simplified/" + condition + str(SR_BIAS) + "/solved_revaluation_one_shot.json", "w") as fp:
            json.dump(solved_revaluation_one_shot/len(SEEDS), fp)
