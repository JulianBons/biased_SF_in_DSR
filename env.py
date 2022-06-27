import numpy as np
import torch

from typing import Union, List, Tuple



class SimpleEnv(object):
    __doc__ = r"""A simple Gridworld Environment consiting of multiple boxes that can be joined together."""

    def __init__(self, 
    height: int, 
    width: int, 
    boundaries: List[List[int]], 
    reward_loc: Union[List[List[int]], List[int]], 
    reward_value: Union[List[int], int], 
    possible_actions: List[List[int]],
    continous_task: bool,
    one_hot: bool) -> None:
        """
        Set up the environment. 
        The dimension of the environment is controlled via the height and width arguments.
        The structure can be set via the boundaries arguments in which boxes are declared as a List of the south-west and north-east corners, which can be joined to form all kinds of structures.
        Finally, the reward states can be set via the reward_loc and reward_values. 
        All possible actions are declared in the possible_actions list.
        """
        
        self.boundaries = boundaries
        self.reward_loc = False
        self.reward_value = 0
        self.change_reward(reward_loc, reward_value)
        self.possible_actions = possible_actions
        self.continous_task = continous_task
        self.one_hot = one_hot


        #Set up the environment
        self.height = height
        self.width = width
        self.env = self.create_env(height, width)
        

    def change_continous(self, continous):
        """
        Params:
            continous -- bool
        """
        self.continous = continous
    
    def change_reward(self, reward_loc: Union[List[Tuple], Tuple], reward_value: Union[List[int], int]) -> None:
        """
        Change the reward state and value
        """
        
        self.reward_loc = reward_loc
        self.reward_value = reward_value


    def create_env(self, height: int, width: int):
        """
        Sets up the internal environment.
        """
      
        #Create the environment defined by the boundaries
        env = np.ones((height, width))*(-1)

        for box in self.boundaries:
            env[box[0][0] : box[1][0]+1, box[0][1] : box[1][1]+1] = 0


        return env


    def move_agent_to_loc(self, agent_loc: Tuple[int]):
        self.agent_loc = agent_loc
        pass



    def coords_within_boundaries(self, coords: Tuple[int], boundaries: Union[List[Tuple], Tuple]) -> bool:
        for box in boundaries:
            if (coords[0] >= box[0][0]) & (coords[0] <= box[1][0]):
                if (coords[1] >= box[0][1]) & (coords[1] <= box[1][1]):
                    return True

        return False
            

    def create_state(self):
        """
        Creates an image of the current state of the environment. 
        If self.one_hot=True, it creates a vector that tabulates the state space.
        """

        if self.one_hot:
            row, col = self.agent_loc
            state = 10*torch.nn.functional.one_hot(torch.tensor([row*self.width + col]), self.height*self.width)
            return state.float()

        else:
            
            state = self.env.copy()
            state[self.agent_loc[0], self.agent_loc[1]] = -10
            return torch.tensor(state).unsqueeze(0).unsqueeze(0).float()



    def create_observation(self, action_idx: int) -> Tuple:
        """
        Observe the effect of the action in the environment
        """

        #Get the current state
        state = self.create_state()
        
        action = self.possible_actions[action_idx]

        reward = 0
        terminal = False

        new_agent_loc = (self.agent_loc[0]+action[0], self.agent_loc[1]+action[1])      

        if isinstance(self.reward_loc, list):
            for i, reward_coord in enumerate(self.reward_loc):
                if isinstance(reward_coord, list):
                    if self.coords_within_boundaries(self.agent_loc, [reward_coord]):
                        reward = self.reward_value[i]
                        terminal = True
                else:
                    if self.agent_loc==reward_coord:
                        reward = self.reward_value[i]
                        terminal = True
        else:
            if self.agent_loc==self.reward_loc:
                reward = self.reward_value
                terminal = True


        if self.coords_within_boundaries(new_agent_loc, self.boundaries):
            self.agent_loc = new_agent_loc
            
        #Get the next state
        next_state = self.create_state()

        #For continous tasks, remove the terminal flag
        if self.continous_task:
            terminal = 0
        

        return (torch.tensor([terminal]), state, action, torch.tensor([reward]), next_state)