import torch
from torch import nn
import numpy as np

import random
from typing import Tuple

import utils

def conv_output_shape(input_size, filter_size, stride, padding):
    return int((input_size + 2*padding - filter_size)/stride + 1)


class DSR_no_encoder(nn.Module):
    __doc__ = r"""Implementaiton of the SF-Agent that takes in the state-vector instead of learning the state-embedding phi."""

    def __init__(self,
                 input_shape: Tuple[int],
                 num_channels: int,
                 num_actions: int,
                 sr_dimensions: int
                 ) -> None:

        super(DSR_no_encoder, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.sr_dimensions = sr_dimensions
        
        output_conv1 = (conv_output_shape(input_shape[0], 3, 1, 1), conv_output_shape(input_shape[1], 3, 1, 1))
        output_conv2 = (conv_output_shape(output_conv1[0], 3, 1, 1), conv_output_shape(output_conv1[1], 3, 1, 1))
        self.output_conv3 = (conv_output_shape(output_conv2[0], 3, 1, 1), conv_output_shape(output_conv2[1], 3, 1, 1))


        #Calculate the reward of the state
        self.reward_values = nn.Linear(sr_dimensions, 1)
        torch.nn.init.normal_(self.reward_values.weight, mean=0.0, std=0.05)
                
                
        #Create the u-network
        self.u_net = nn.Sequential(
                nn.Conv1d(in_channels=sr_dimensions*num_actions, out_channels=sr_dimensions//2 *num_actions, kernel_size=1, groups=num_actions),
                nn.ReLU(),
                nn.Conv1d(sr_dimensions//2 *num_actions, sr_dimensions*num_actions, kernel_size=1, groups=num_actions),
                )


    def forward(self, state):     
        
        #Create the state embedding phi
        state_embedding_detached = state.detach()

        #Estimate the imediate reward
        reward_values = self.reward_values(state)

        #Get successor features
        successor_features = self.u_net(state_embedding_detached.unsqueeze(-1).repeat(1, self.num_actions, 1))
        successor_features = successor_features.reshape(-1, self.num_actions, self.sr_dimensions)


        return state_embedding_detached, reward_values, successor_features



class DSR_agent_no_encoder(object):
    
    def __init__(self, info):

        self.current_step = 0
        self.device = info['device']
        self.channel = info['channel']
        self.height = info['height']
        self.width = info['width']
        self.batch_size = info['batch_size']
        self.num_actions = info['num_actions']
        self.sr_dimensions = info['sr_dimensions']
        
        #Set up hyperparameter
        self.gamma = info['gamma']
        self.lr = info['lr']
        self.eps = info['eps']
        self.weight_decay = info['weight_decay']
        self.strategy = info['strategy']
        self.sr_bias = info['sr_bias']
        self.reward_weight = info['reward_weight']
        self.decoder_weight = info['decoder_weight']
        self.sr_weight = info['sr_weight']
        self.criterion_no_clip = nn.MSELoss(reduction='sum')
        self.criterion_clip = nn.HuberLoss(reduction="sum")

        #Set up the networks
        self.policy_network = DSR_no_encoder((self.height, self.width), self.channel, self.num_actions, self.sr_dimensions).to(self.device)
        self.target_network = DSR_no_encoder((self.height, self.width), self.channel, self.num_actions, self.sr_dimensions).to(self.device)

        #Set up the optimizer
        self.optimizer = torch.optim.RMSprop(params=self.policy_network.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)


        #Copy the weights of the policy network to the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()


    @torch.no_grad()
    def get_w(self):
        return self.policy_network.reward_values.weight.data

    @torch.no_grad()
    def set_w(self, REWARD_LOC, REWARD_VALUES):
        self.policy_network.reward_values.weight = nn.Parameter(torch.zeros_like(self.policy_network.reward_values.weight.data))

        for i,loc in enumerate(REWARD_LOC):
            
            self.policy_network.reward_values.weight[0, loc] = float(REWARD_VALUES[i])

        self.policy_network.reward_values.weight.requires_grad = False
        self.policy_network.reward_values.bias.requires_grad = False


    @torch.no_grad()
    def get_q_values(self, state, network):
            
        w = network.reward_values.weight.data

        state = state.to(self.device)

        _, _, sr_m  = network(state)

        q_values = sr_m @ w.T

        return q_values

    @torch.no_grad()
    def select_action_epsilon_greedy(self, state):
        epsilon_rate = self.strategy.getEpsilon(self.current_step)
        self.current_step += 1

        if epsilon_rate > random.random():
            action = random.randint(0, self.num_actions-1)
            return torch.tensor([action])

        else:
            #Get action
            q_values = self.get_q_values(state, self.policy_network)

            action = torch.argmax(q_values, dim=1)

            return action

    @torch.no_grad()
    def select_action_greedy(self, state):
        #Get action
        q_values = self.get_q_values(state, self.policy_network)
        action = torch.argmax(q_values, dim=1)
        return action

    def reset_epsilon(self):
        self.current_step = 0
        pass

    def change_loss(self, reward_weight, sr_weight, decoder_weight):
        self.reward_weight = reward_weight
        self.sr_weight = sr_weight
        self.decoder_weight = decoder_weight
        pass

    

    def train(self, observation):
        """
        Train the policy network
        """
        is_terminal, states, actions, rewards, next_states = utils.extract_tensors(observation)

        #Process the observation with the policy network
        state_embedding, reward_vals, u_features = self.policy_network(states)

        with torch.no_grad():
            #Get the w weights of the target network
            w_target = self.target_network.reward_values.weight.data

            #Process the observation with the target network
            state_embedding_target, _, u_features_target = self.target_network(next_states)

            #Get q-values
            q_vals_target = u_features_target @ w_target.T

            #Get the action that maximizes the q-values
            action_argmax = torch.argmax(q_vals_target, dim=1).repeat(1, self.sr_dimensions).unsqueeze(1)

            #Get the u features for the most valuable action
            successor_feat_target_bias = torch.gather(u_features_target, 1, action_argmax).reshape(self.batch_size, self.sr_dimensions)

            successor_feat_target_unbiased = torch.mean(u_features_target, dim=1)
            
            successor_feat_target = (1-self.sr_bias)*successor_feat_target_unbiased + (self.sr_bias)*successor_feat_target_bias 

        #Get the predicted successor feature for the next state
        successor_feat = torch.gather(u_features, 1, actions.unsqueeze(1).repeat(1, self.sr_dimensions).unsqueeze(1))

        ##Define the losses
        reward_loss = self.criterion_clip(reward_vals.reshape(-1).float(), rewards.float())

        #For the successor_features
        u_loss = self.criterion_clip(successor_feat.reshape(self.batch_size, self.sr_dimensions).float(), state_embedding_target.float() + self.gamma*successor_feat_target.reshape(self.batch_size, self.sr_dimensions).float())

        ##Combine the losses
        loss = self.reward_weight*reward_loss + self.sr_weight*u_loss
        

        self.optimizer.zero_grad()

        loss.backward(retain_graph=True)

        self.optimizer.step()


        return (loss.item(), reward_loss.item(), u_loss.item())


    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())


    def freeze_layers(self):
        for param in self.policy_network.parameters():
            param.requires_grad = False

        self.policy_network.reward_values.weight.requires_grad = True
        self.policy_network.reward_values.bias.requires_grad = True

    def reset_optimizer(self):
        self.optimizer = torch.optim.RMSprop(params=self.policy_network.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)







####End-to-End
class DSR_end_to_end(nn.Module):
    __doc__ = r"""Implementation of the End-to-End DSR"""

    def __init__(self,
                 input_shape: Tuple[int],
                 num_channels: int,
                 num_actions: int,
                 sr_dimensions: int
                 ) -> None:

        super(DSR_agent_end_to_end, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.sr_dimensions = sr_dimensions


        ####Define the architecture####
        #Encode the state
        self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
               )
        
        output_conv1 = (conv_output_shape(input_shape[0], 3, 1, 1), conv_output_shape(input_shape[1], 3, 1, 1))
        output_conv2 = (conv_output_shape(output_conv1[0], 3, 2, 1), conv_output_shape(output_conv1[1], 3, 2, 1))
        self.output_conv3 = (conv_output_shape(output_conv2[0], 3, 1, 1), conv_output_shape(output_conv2[1], 3, 1, 1))


        #Create the successor features for the state
        self.state_features = nn.Sequential(
                nn.Linear(np.prod(self.output_conv3)*64, sr_dimensions//2),
                nn.ReLU(),
                nn.Linear(sr_dimensions//2, sr_dimensions),
                )


        #Calculate the reward of the state
        self.reward_values = nn.Linear(sr_dimensions, 1)
        torch.nn.init.normal_(self.reward_values.weight, mean=0.02, std=0.01)


        #Create the decoder
        self.decoder_input = nn.Sequential(
                nn.Linear(sr_dimensions, 64*np.prod(self.output_conv3))
                )


        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
                )
                
                
        #Create the u-network
        self.u_net = nn.Sequential(
                nn.Conv1d(in_channels=sr_dimensions*num_actions, out_channels=sr_dimensions//2 *num_actions, kernel_size=1, groups=num_actions),
                nn.ReLU(),
                nn.Conv1d(sr_dimensions//2 *num_actions, sr_dimensions*num_actions, kernel_size=1, groups=num_actions)
                )


    def forward(self, state):

        #Encode the state
        encoding = torch.flatten(self.encoder(state), start_dim=1)
        state_embedding = self.state_features(encoding)    
        
        #Create the state embedding phi
        state_embedding_detached = state_embedding.detach()

        #Estimate the imediate reward
        reward_values = self.reward_values(state_embedding)

        #Decode the state embedding
        decoder_input = self.decoder_input(state_embedding).reshape(-1, 64, self.output_conv3[0], self.output_conv3[1])
        decoder = self.decoder(decoder_input)

        #Get successor features
        successor_features = self.u_net(state_embedding_detached.unsqueeze(-1).repeat(1, self.num_actions, 1))
        successor_features = successor_features.reshape(-1, self.num_actions, self.sr_dimensions)


        return state_embedding_detached, reward_values, decoder, successor_features



class DSR_agent_end_to_end(object):
    
    def __init__(self, info):

        self.current_step = 0
        self.device = info['device']
        self.channel = info['channel']
        self.height = info['height']
        self.width = info['width']
        self.batch_size = info['batch_size']
        self.num_actions = info['num_actions']
        self.sr_dimensions = info['sr_dimensions']
        
        #Set up hyperparameter
        self.gamma = info['gamma']
        self.lr = info['lr']
        self.eps = info['eps']
        self.weight_decay = info['weight_decay']
        self.strategy = info['strategy']
        self.sr_bias = info['sr_bias']
        self.reward_weight = info['reward_weight']
        self.decoder_weight = info['decoder_weight']
        self.sr_weight = info['sr_weight']
        self.criterion_no_clip = nn.MSELoss(reduction='sum')
        self.criterion_clip = nn.HuberLoss(reduction="sum")

        #Set up the networks
        self.policy_network = DSR_end_to_end((self.height, self.width), self.channel, self.num_actions, self.sr_dimensions).to(self.device)
        self.target_network = DSR_end_to_end((self.height, self.width), self.channel, self.num_actions, self.sr_dimensions).to(self.device)

        #Set up the optimizer
        self.optimizer = torch.optim.RMSprop(params=self.policy_network.parameters(), lr=self.lr)


        #Copy the weights of the policy network to the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()



    def reset_optimizer(self):
        self.optimizer = torch.optim.RMSprop(params=self.policy_network.parameters(), lr=self.lr)



    @torch.no_grad()
    def get_w(self):
        return self.policy_network.reward_values.weight.data

    @torch.no_grad()
    def get_q_values(self, state, network):
            
        w = network.reward_values.weight.data

        state = state.to(self.device)

        _, _, _, sr_m  = network(state)

        q_values = sr_m @ w.T

        return q_values

    @torch.no_grad()
    def select_action_epsilon_greedy(self, state):
        epsilon_rate = self.strategy.getEpsilon(self.current_step)
        self.current_step += 1

        if epsilon_rate > random.random():
            action = random.randint(0, self.num_actions-1)
            return torch.tensor([action])

        else:
            #Get action
            q_values = self.get_q_values(state, self.policy_network)

            action = torch.argmax(q_values, dim=1)

            return action

    @torch.no_grad()
    def select_action_greedy(self, state):
        #Get action
        q_values = self.get_q_values(state, self.policy_network)
        action = torch.argmax(q_values, dim=1)
        return action

    def reset_epsilon(self):
        self.current_step = 0
        pass

    def change_loss(self, reward_weight, sr_weight, decoder_weight):
        self.reward_weight = reward_weight
        self.sr_weight = sr_weight
        self.decoder_weight = decoder_weight

        pass

    

    def train(self, observation):
        """
        Train the policy network
        """
        is_terminal, states, actions, rewards, next_states = utils.extract_tensors(observation)

        #Process the observation with the policy network
        state_embedding, reward_vals, decoder, u_features = self.policy_network(states)

        with torch.no_grad():
            #Get the w weights of the target network
            w_target = self.target_network.reward_values.weight.data

            #Process the observation with the target network
            state_embedding_target, _, _, u_features_target = self.target_network(next_states)

            #Get q-values
            q_vals_target = u_features_target @ w_target.T

            #Get the action that maximizes the q-values
            action_argmax = torch.argmax(q_vals_target, dim=1).repeat(1, self.sr_dimensions).unsqueeze(1)

            #Get the u features for the most valuable action
            successor_feat_target_bias = torch.gather(u_features_target, 1, action_argmax).reshape(self.batch_size, self.sr_dimensions)

            successor_feat_target_unbiased = torch.mean(u_features_target, dim=1)

            
            successor_feat_target = (1-self.sr_bias)*successor_feat_target_unbiased + (self.sr_bias)*successor_feat_target_bias


        #Get the predicted successor feature for the next state
        successor_feat = torch.gather(u_features, 1, actions.unsqueeze(1).repeat(1, self.sr_dimensions).unsqueeze(1))

        ##Define the losses
        reward_loss = self.criterion_clip(reward_vals.reshape(-1).float(), rewards.float())

        #For the successor_features
        u_loss = self.criterion_clip(successor_feat.reshape(self.batch_size, self.sr_dimensions).float(), state_embedding_target.float() + self.gamma*successor_feat_target.reshape(self.batch_size, self.sr_dimensions).float())


        #For the Decoder
        decoder_loss = self.criterion_no_clip(decoder.float(), states.float())

        ##Combine the losses
        loss = self.reward_weight*reward_loss + self.sr_weight*u_loss + self.decoder_weight*decoder_loss 
        

        self.optimizer.zero_grad()

        loss.backward(retain_graph=True)

        self.optimizer.step()


        return (loss.item(), reward_loss.item(), u_loss.item())


    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())


    def freeze_layers(self):
        for param in self.policy_network.parameters():
            param.requires_grad = False

        self.policy_network.reward_values.weight.requires_grad = True
        self.policy_network.reward_values.bias.requires_grad = True

