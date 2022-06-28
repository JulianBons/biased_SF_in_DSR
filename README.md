# Model-based Learning with the Deep Successor Representation

This repository contains the code of the biased-SF DSR used in the experiments. 

The agents.py file contains the Pytorch implementation of the DSR. The hyperparameter $\omega_{bias}$ controls the degree to which the SF is biased towards the optimal policy. 
Setting 
$\omega_{bias}=1$
fully overfits on the optimal-policy of the given task, while
$\omega_{bias}=0$
reflects the environment fully independent of the task. Setting
$\omega_{bias}$
in between these two values allows to trade off flexibility with functional range.

The env.py file contains the implementation of the small Gridworld task. 

The simulations of the latent learning task can be found in latent_learning.py and the revaluation task is implemented in revaluation.py
