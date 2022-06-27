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
REWARD_LOC = [(2, 2), (1, 1)]
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


states_to_encode = []
for box in BOUNDARIES:
    for dim1 in range(box[0][0], box[1][0]+1):
        for dim2 in range(box[0][1], box[1][1]+1):
            states_to_encode.append((dim1, dim2))


if not os.path.isdir("Latent_Learning_simplified/"):
    os.mkdir("Latent_Learning_simplified/")

for SR_BIAS in np.arange(0, 1.01, 0.05):

    for NUM_EXPLORATION_STEPS in [1, 10000]: 
    
      print(f"STARTING EXPERIMENT WITH SR_BIAS: {SR_BIAS} and NUM_EXPLORATION_STEPS: {NUM_EXPLORATION_STEPS}")

      for seed in [58, 39, 14, 192, 491, 18, 501, 59, 48, 317, 49, 318, 1283, 4812, 139, 57, 581, 128]:
        

            if not os.path.isdir("Latent_Learning_simplified/"+ str(SR_BIAS)):
                os.mkdir("Latent_Learning_simplified/"+ str(SR_BIAS))

            if not os.path.isdir("Latent_Learning_simplified/" + str(SR_BIAS) + "/" + str(NUM_EXPLORATION_STEPS)):
                os.mkdir("Latent_Learning_simplified/"+ str(SR_BIAS) + "/"+ str(NUM_EXPLORATION_STEPS))


            torch.manual_seed(seed)
            random.seed(seed)

            SR_WEIGHT = 1-REWARD_WEIGHT 


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

            #Exploration Phase
            env.change_reward(reward_loc=REWARD_LOC, reward_value=[0, 0])
            env.move_agent_to_loc(random.choice(states_to_encode))
            env.change_continous(True)
            rewards_over_time = []
            current_steps = 0

            #agent.set_w([(8*WIDTH + 2), (8*WIDTH + 8)], [0, 0])

            for step in range(1, NUM_EXPLORATION_STEPS):

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
                    #Reset condition
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



                                    




            ###TASK Phase
            env.change_reward(reward_loc=(8, 8), reward_value=2)
            env.change_continous(False)
            rewards_over_time = []

            memory = utils.ReplayMemory(MEMORY_SIZE)


            agent.set_w([8*WIDTH+8], [2])

            current_step = 0
            agent.reset_epsilon()
            env.move_agent_to_loc(START_LOC)

            agent.reset_optimizer()

            episodes_needed = 0
            found_reward = 0


            for step in range(25000):

                current_steps += 1

                state = env.create_state().to(device)
                action_idx = agent.select_action_epsilon_greedy(state)

                terminal, state, action, reward, next_state = env.create_observation(action_idx)

                terminal = terminal.to(device); state = state.to(device); action_idx = action_idx.reshape(-1).to(device); reward = reward.to(device); next_state = next_state.to(device)

                if torch.any(state != next_state):
                    memory.push(observation(terminal, state, action_idx, reward, next_state))

                if terminal:
                    memory.push(observation(terminal, state, action_idx, reward, next_state))
                    env.move_agent_to_loc(START_LOC)
                    current_step = 0

                if current_steps%EPISODE_LENGTH==0:
                    env.move_agent_to_loc(START_LOC)
                    current_steps = 0
                
                if (step>MEMORY_SIZE//10) & (step%2==0):
                    observations = memory.sample(BATCH_SIZE)
                    (loss, reward_loss, u_loss) = agent.train(observations)

                if (step%100==0) & (step>MEMORY_SIZE//10):
                    losses.append(loss)

                if step%TARGET_UPDATE==0:
                    agent.update_target_network()

                if (step%25==0):
                    
                    env_sim = copy.copy(env)
                    reward_count = 0
                    states_visited = []

                    episodes_needed+=1

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

                        if reward_count>0:
                            found_reward+=1
                        
                if found_reward==5:
                    break
            

            with open("Latent_Learning_simplified/"+ str(SR_BIAS) + "/"+ str(NUM_EXPLORATION_STEPS) + "/episodes_experiment_"+str(seed)+".json", "w") as fp:
                json.dump(episodes_needed*25, fp)
                        
 