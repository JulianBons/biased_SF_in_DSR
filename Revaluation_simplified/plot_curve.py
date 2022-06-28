import sys
import os
import json

import matplotlib.pyplot as plt


plt.figure(figsize=(6, 5.5))
colors = ["firebrick", "slategrey"]
labels = ["distant", "close"]

conditions = [condition for condition in os.listdir() if os.path.isdir(condition)]

for i, condition in enumerate(conditions):
    weight_biases = sorted([bias for bias in os.listdir(condition) if os.path.isdir(os.path.join(condition, bias))])
    
    revaluation_score = []

    x = [float(bias) for bias in weight_biases]

    for bias in weight_biases:
        with open(os.path.join(condition, bias) + "/times_changed.json", "r") as fp:
            revaluation_score.append(json.load(fp))

    plt.plot(x, revaluation_score, color=colors[i], linewidth=4)

plt.scatter([None], [None], color=colors[1], label=labels[1], s=70, edgecolors="none")
plt.scatter([None], [None], color=colors[0], label=labels[0], s=70, edgecolors="none")

plt.ylabel("Number of steps")
plt.xlabel("Episode")
lgnd = plt.legend(frameon=False, fontsize=15, loc=(0.7, 0.6), handletextpad=0.1)

plt.axis("off")
plt.text(-0.06, 1, "1", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.09, 0.5, "0.5", fontsize=15, alpha=0.85, verticalalignment="center")
plt.text(-0.06, 0, "0", fontsize=15, alpha=0.85,verticalalignment="center") 

plt.text(-0.17, 0.5, "% Change in Preference", fontsize=17, rotation="vertical", verticalalignment="center")
plt.text(0.5, -0.15, r"$\omega_{bias}$", fontsize=17, horizontalalignment="center")


#plt.text(0.05, -0.07, "0.05", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(0.225, -0.07, "0.25", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(0.475, -0.07, "0.5", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(0.7255, -0.07, "0.75", fontsize=15, alpha=0.85, horizontalalignment="center")
plt.text(1, -0.07, "1", fontsize=15, alpha=0.85, horizontalalignment="center")

plt.hlines(1, 0, 1, color="black", linestyles="--", alpha=0.3)
plt.hlines(0.5, 0, 1, color="black", linestyles="--", alpha=0.3)
plt.hlines(0, 0, 1, color="black", linestyles="--", alpha=0.3)
plt.vlines(0.025, 0, 1.025, color="firebrick", alpha=0.25)
plt.vlines(0.1, 0, 1.025, color="firebrick", alpha=0.25)

plt.fill_between([0.025, 0.1], 0, 1.025, color="firebrick", alpha=0.2)
plt.savefig("bias_revaluation.png", format="png", dpi=1200)


plt.show()
