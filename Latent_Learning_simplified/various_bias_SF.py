import os
import json

import numpy as np
import matplotlib.pyplot as plt

def sortkey(file):
    return float(file)

sr_biases = sorted([bias for bias in os.listdir() if os.path.isdir(bias)], key=sortkey)


def prepare_array(files):
    simulation_results = []
    for file in files:
        with open(file, "r") as fp:
            number_of_episodes_needed = json.load(fp)
            
        simulation_results.append(number_of_episodes_needed)

    experimental_result = np.mean(simulation_results)

    return experimental_result
            

result_exploration_condition = []

fig = plt.figure(figsize=(6, 1.9))

x = [float(bias) for bias in sr_biases]

for sr_bias in sr_biases:
    path = os.path.join(sr_bias, "10000")
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")]

    result_sr_bias = prepare_array(files)

    result_exploration_condition.append(result_sr_bias)

plt.plot(x, result_exploration_condition, color="firebrick", linewidth=4)



plt.text(-0.06, 0, "0", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.09, 300, "30", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.09, 600, "60", fontsize=15, alpha=0.85, verticalalignment="center")

plt.hlines(0, 0, 1, color="black", linestyle="--", alpha=0.3)
plt.hlines(300, 0, 1, color="black", linestyle="--", alpha=0.3)
plt.hlines(600, 0, 1, color="black", linestyle="--", alpha=0.3)


plt.vlines(0.025, 0, 625, color="firebrick", alpha=0.25)
plt.vlines(0.1, 0, 625, color="firebrick", alpha=0.25)

plt.fill_between([0.025, 0.1], 0, 625, color="firebrick", alpha=0.2)

plt.axis("off")

plt.savefig("learning_episodes_with_exploration.png", format="png", dpi=1200)

plt.show()

plt.close()


fig = plt.figure(figsize=(6, 2.9))

result_exploration_condition = []
for sr_bias in sr_biases:
    path = os.path.join(sr_bias, "1")
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")]

    result_sr_bias = prepare_array(files)

    result_exploration_condition.append(result_sr_bias)

plt.plot(x, result_exploration_condition, color="slategrey", linewidth=4)

#plt.ylim(0, 650)

plt.text(-0.12, 1400, "140", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.12, 1700, "170", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.12, 2000, "200", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.12, 2300, "230", fontsize=15, alpha=0.85, verticalalignment="center")

plt.hlines(1400, 0, 1, color="black", linestyle="--", alpha=0.3)
plt.hlines(1700, 0, 1, color="black", linestyle="--", alpha=0.3)
plt.hlines(2000, 0, 1, color="black", linestyle="--", alpha=0.3)
plt.hlines(2300, 0, 1, color="black", linestyle="--", alpha=0.3)


plt.vlines(0.025, 1350, 2350, color="firebrick", alpha=0.25)
plt.vlines(0.1, 1350, 2350, color="firebrick", alpha=0.25)

plt.fill_between([0.025, 0.1], 1350, 2350, color="firebrick", alpha=0.2)

plt.axis("off")

plt.savefig("learning_episodes_from_scratch.png", format="png", dpi=1200)

plt.show()
