import os
import json

import numpy as np
import matplotlib.pyplot as plt

sr_biases = [bias for bias in os.listdir() if os.path.isdir(bias)]

def prepare_array(files):

    experiment_result = np.zeros(10000)
    for file in files:
        simulation_result = np.zeros(10000)
        with open(file, "r") as fp:
            number_of_episodes_needed = json.load(fp)
            
        simulation_result[number_of_episodes_needed:] = 1

        experiment_result += simulation_result

    return experiment_result/len(files)
            

for exploration_condition in ["1", "10000"]:

    files = [os.path.join(exploration_condition, file) for file in os.listdir(exploration_condition) if file.endswith(".json")]

    result_sr_bias = prepare_array(files)



    plt.plot(range(len(result_sr_bias)), result_sr_bias, label=exploration_condition)

plt.legend()
plt.show()



