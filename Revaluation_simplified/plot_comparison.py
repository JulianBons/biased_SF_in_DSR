import sys
import os
import json

import matplotlib.pyplot as plt
import numpy as np


colors = "firebrick slategrey".split()


unbiased_off_policy = []
biased_on_policy = []

conditions = "close distal".split()
labels = "unbiased optimal".split()

for i, condition in enumerate(conditions):

    with open(os.path.join(condition, "0.0") + "/times_changed.json", "r") as fp:
        unbiased_off_policy.append(json.load(fp))

    with open(os.path.join(condition, "1.0") + "/times_changed.json", "r") as fp:
        biased_on_policy.append(json.load(fp))


x = np.arange(len(conditions))
width = 0.4

fig, ax = plt.subplots(figsize=(6,6))
rects1 = ax.bar(x - width/2-0.0125, [y+0.01 for y in  unbiased_off_policy], width, color="firebrick")
rects2 = ax.bar(x + width/2+0.0125, [y+0.01 for y in biased_on_policy], width, color="slategrey")

plt.scatter([None], [None], color=colors[0], label=labels[0], s=70, edgecolors="none")
plt.scatter([None], [None], color=colors[1], label=labels[1], s=70, edgecolors="none")


plt.legend(frameon=False, fontsize=15, loc=(0.6, 0.6), handletextpad=0.1)

unbiased = [max(height-0.07, 0.035) for height in unbiased_off_policy]
biased = [max(height-0.07, 0.035) for height in biased_on_policy]

unbiased_color = []
biased_color = []

for val in unbiased_off_policy:
    if val==0:
        unbiased_color.append("black")
    else:
        unbiased_color.append("white")

for val in biased_on_policy:
    if val==0:
        biased_color.append("black")
    else:
        biased_color.append("white")


plt.text(-width/2 - 0.0125, unbiased[0], str(unbiased_off_policy[0]), horizontalalignment="center", fontsize=22, color=unbiased_color[0])
plt.text(width/2+0.0125, biased[0], str(biased_on_policy[0]), horizontalalignment="center", fontsize=22, color=biased_color[0])
plt.text(1-width/2 - 0.0125, unbiased[1], str(unbiased_off_policy[1]), horizontalalignment="center", fontsize=22, color=unbiased_color[1])
plt.text(1+width/2+0.0125, biased[1], str(biased_on_policy[1]), horizontalalignment="center", fontsize=22, color=biased_color[1])

plt.text(0, -0.07, "Close", horizontalalignment="center", fontsize=15, alpha=0.85)
plt.text(1, -0.07, "Distant", horizontalalignment="center", fontsize=15, alpha=0.85)

plt.text(-0.65, 0.5, "% Change in Preference", fontsize=17, rotation="vertical", verticalalignment="center")
plt.text(0.5, -0.15, "Condition", fontsize=17, horizontalalignment="center")

plt.axis("off")

plt.savefig("revaluation_comparison.png", format="png", dpi=1200)

plt.show()
