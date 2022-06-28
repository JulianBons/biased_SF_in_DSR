import os
import json

import numpy as np
import matplotlib.pyplot as plt

label = ["unbiased", "optimal"]
colors = ["firebrick", "slategrey"]


linestyle = [(0, (0.5, 0.5)), "-"]
alphas = [0.85, 1]


def prepare_array(files):

    experiment_result = np.zeros(10000)
    for file in files:
        simulation_result = np.zeros(10000)
        with open(file, "r") as fp:
            number_of_episodes_needed = json.load(fp)
            
        simulation_result[number_of_episodes_needed:] = 1

        experiment_result += simulation_result

    return experiment_result/len(files)


plt.figure(figsize=(6, 6))

for sr_bias in ["0.0", "1.0"]:

    for i, exploration_condition in enumerate(["1", "10000"]):

        path = os.path.join(sr_bias, exploration_condition)

        files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith(".json")]

        result_sr_bias = prepare_array(files)



        plt.plot(range(len(result_sr_bias)), result_sr_bias, color=colors[int(float(sr_bias))], linestyle=linestyle[i], linewidth=4, alpha=alphas[i])

plt.scatter([None], [None], color=colors[0], s=60, edgecolors="none", label=label[0])
plt.scatter([None], [None], color=colors[1], s=60, edgecolors="none", label=label[1])

plt.plot([None], linestyle=linestyle[1], label="After exploration", color="black")
plt.plot([None], linestyle=linestyle[0], label="From scratch", color="black")


plt.text(-650, 0.5, "%i Agents solved task", fontsize=17, rotation="vertical", verticalalignment="center")

plt.text(-200, 1, "1", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-350, 0.5, "0.5", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-200, 0, "0", fontsize=15, alpha=0.85, verticalalignment="center")

plt.hlines(1, 0, 10000, color="black", linestyles="--", alpha=0.3)
plt.hlines(0.5, 0, 10000, color="black", linestyles="--", alpha=0.3)
plt.hlines(0, 0, 10000, color="black", linestyles="--", alpha=0.3)


plt.text(2500, -0.15, "Episode", fontsize=17, horizontalalignment="center")

plt.text(1250, -0.07, "10", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(2500, -0.07, "20", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(3750, -0.07, "30", fontsize=15, alpha=0.85, horizontalalignment="center")




plt.xlim(0, 5000)
plt.axis("off")



plt.legend(frameon=False, fontsize=15, handletextpad=1, loc=(0.5, 0.175))

plt.savefig("steps_needed_unbiased_versus_optimal.png", format="png", dpi=1200)

plt.show()



