import torch
import random

from collections import namedtuple

observation = namedtuple('observation', ('terminal', 'state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    def push(self, observation):
        if len(self.memory) < self.capacity:
            self.memory.append(observation)
        else:
            self.memory[self.push_count%self.capacity] = observation
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) > batch_size


class EpsilonGreedyStrategy(object):

    def __init__(self, start, end, decay_over_steps):
        self.start = start
        self.end = end
        self.decay_over_steps = decay_over_steps

    def getEpsilon(self, current_step):

        return self.start - current_step * (self.start-self.end)/self.decay_over_steps 


def extract_tensors(observations):
    batch = observation(*zip(*observations))

    states = torch.cat(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.cat(batch.next_state)
    terminal = torch.cat(batch.terminal)
    

    states = states.float()
    next_states = next_states.float()
   
    return (terminal, states, actions, rewards, next_states)