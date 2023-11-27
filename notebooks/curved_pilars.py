#%% 1 setup
import copy
from os import path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

modules_path = path.join(path.dirname(__file__), '../src')
sys.path.append(modules_path)
from signal_simulation.signal_generator import simulate_single_signal

CURRENT_DIR = path.dirname(__file__)
OLD_NOISE_NOFLAT_SCALES = path.join(CURRENT_DIR, '../assets/training-databases/old_noise-no_flat-scales.json')
# try old_noise-no_flat to see a bigger curved pillar effect
OG_RENAMED = path.join(CURRENT_DIR, '../assets/training-databases/original-renamed.json')
data = pd.read_json(OLD_NOISE_NOFLAT_SCALES)
og = pd.read_json(OG_RENAMED)

#%% 2 Computing Jdoubling successes
def max_aceptado (x):
    return x * 0.95

def min_aceptado (x):
    return x * 1.05

def add_cond_columns(database):
    database["Max_accept"] = database["theoretical_J"].apply(max_aceptado)
    database["Min_accept"] = database["theoretical_J"].apply(min_aceptado)
    conds = [(database.calculated_J >= database.Max_accept) & (database.calculated_J <= database.Min_accept), 
                   (database.calculated_J < database.Max_accept) & (database.calculated_J > database.Min_accept)]
    elecciones = np.array((0, 1), dtype="int8")
    database["Cluster"] = np.select(conds, elecciones, -1)

add_cond_columns(data)

#%% 2 pillars
plt.scatter(data["signal_noise_ratio"], data["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], data["Cluster"]))
plt.show()

#%% 3 investigation
# The original distributed values
#signal_width_values = np.arange(0.5, 24.0, 0.5)
#theoretical_J_values = np.arange(0.5, 12.0, 0.05)

section = data.loc[abs(data["calculated_J"] - 7.0) <= 0.1]
plt.scatter(section["signal_noise_ratio"], section["signal_width"], s=2)
plt.show()

lows = section.loc[section["signal_width"] <= 1.0]
highs = section.loc[section["signal_width"] > 1.0]

plt.scatter(data["signal_noise_ratio"], data["calculated_J"], s=2, c='#303030')
plt.scatter(lows["signal_noise_ratio"], lows["calculated_J"], s=2, c='#ff2020')
plt.scatter(highs["signal_noise_ratio"], highs["calculated_J"], s=2, c='#ffff00')
plt.show()

#%% 4 boundaries

def dist(x ,y):
    return abs(x-y)

def cluster_bounds():
    "find the bounds for the clusters without using any heuristic"
    snrs = list(lows["signal_noise_ratio"])
    snrs = sorted(snrs)

    directions = []
    for (i, x) in enumerate(snrs):
        if i == len(snrs) - 1:
            directions.append(-1)
        elif i == 0:
            directions.append(1)
        else:
            r = snrs[i+1]
            l = snrs[i-1]
            directions.append(-1) if dist(l,x) < dist(r,x) else directions.append(1)

    bounds = []
    for i in range(len(directions)-2):
        d = directions[i]
        if d == -1 and directions[i+1] == 1:
            bounds.append(snrs[i])
            #bounds.append(snrs[i+1])
    return bounds


bounds = cluster_bounds() 
plt.scatter(data["signal_noise_ratio"], data["calculated_J"], s=2, c='#303030')
plt.scatter(lows["signal_noise_ratio"], lows["calculated_J"], s=2, c='#ff2020')
for b in bounds:
    plt.axvline(x=b, c='#ff00ff')
plt.show()

print(f"{len(bounds)} pivots")
print(f"last ten pivots: {bounds[-10:]}")

#%% 5 distribution of snr by itself
plt.eventplot(data["signal_noise_ratio"])
plt.show()
plt.eventplot(data["noise_scale"])
plt.show()

plt.scatter(data["noise_scale"], data["signal_noise_ratio"])
plt.show()

plt.scatter(section["noise_scale"], section["signal_noise_ratio"])
plt.show()

#%% 6 sanity check
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data["noise_scale"], data["signal_width"], data["theoretical_J"], marker='^', s=0.1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.view_init(45,20,40)
ax.view_init(75,30,0)
ax.scatter(data["noise_scale"], data["signal_noise_ratio"], data["theoretical_J"], marker='^', s=0.1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data["noise_scale"], data["signal_noise_ratio"], data["theoretical_J"], marker='^', s=0.1)
ax.view_init(90,30,0)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data["noise_scale"], data["signal_noise_ratio"], data["theoretical_J"], marker='^', s=0.1)
ax.view_init(0,0,0)
plt.show()
# note: %matplotlib widget <- will trigger interactive 3d but behaves strangely with the rest of the notebook
# What this last 3d graph shows is that indeed there are 10 "pillars"
# each corresponding to one of the ten available values choosen when we simulate for noise_scale.
# These noise scale values produce three regions:
# 1 a diagonal compared with snr, what we originally expected. Good noise_scale => good snr
# 2 another diagonal, good noise_scale => decent snr
# 3 triangle below 2, here the snr only seems bound by 2 and there doesn't seem to be a more complex relationship
#
# When we plot this without the noise scale axis, the "pillars" in 2 and 3 overlap wich makes it seem as if there
# were more pillars than noise_scales.
#
# I predict that adding more resolution to noise_scales will make the graph seem more uniform.
# Though I must say that we do not have an explanation for why these 3 regions are formed in the first place

#%% 7 what dat width do ?
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data["noise_scale"], data["signal_noise_ratio"], data["signal_width"], marker='^', s=0.1)
ax.view_init(45,45,45)
plt.show()
#%% 8 Does the bigger dataset "erase" the columns
BIGGER_DATASET = path.join(CURRENT_DIR, '../assets/training-databases/old_noise-no_flat-scales-bigger.json')
bigger = pd.read_json(BIGGER_DATASET)
add_cond_columns(bigger)
TUNNED_DATASET = path.join(CURRENT_DIR, '../assets/training-databases/morescales-lesswidths.json')
tunned = pd.read_json(TUNNED_DATASET)
add_cond_columns(tunned)

plt.scatter(data["signal_noise_ratio"], data["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], data["Cluster"]))
plt.show()
plt.scatter(bigger["signal_noise_ratio"], bigger["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], bigger["Cluster"]))
plt.show()
plt.scatter(tunned["signal_noise_ratio"], tunned["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], tunned["Cluster"]))
plt.show()