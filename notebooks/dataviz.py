#%% 1 database setup
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = path.dirname(__file__)
DB_DIR = path.join(CURRENT_DIR, "../assets/training-databases/")

ORIGINAL = path.join(DB_DIR, 'original-renamed.json')
OLD_NOISE_FLAT = path.join(DB_DIR, 'old_noise-flat.json')
OLD_NOISE_NOFLAT = path.join(DB_DIR, 'old_noise-no_flat-scales.json')
NEW_NOISE_FLAT = path.join(DB_DIR, 'new_noise-flat.json')
OLD_CURVED_FLAT = path.join(DB_DIR, 'old-curved-flat.json')
OLD_FLAT_PREVSNR = path.join(DB_DIR, 'old-flat-prevsnr.json')

db_original = pd.read_json(ORIGINAL)
db_of = pd.read_json(OLD_NOISE_FLAT)
db_o = pd.read_json(OLD_NOISE_NOFLAT)
db_nf= pd.read_json(NEW_NOISE_FLAT)
db_ocf=pd.read_json(OLD_CURVED_FLAT)
db_ofp=pd.read_json(OLD_FLAT_PREVSNR)

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

add_cond_columns(db_original)
add_cond_columns(db_of)
add_cond_columns(db_o)
add_cond_columns(db_nf)
add_cond_columns(db_ocf)
add_cond_columns(db_ofp)


#%% 3 Plotting input parameters
f, ax = plt.subplots(5)
ax[0].scatter(db_original["theoretical_J"], db_original["signal_width"], s=2)
ax[1].scatter(db_of["theoretical_J"], db_of["signal_width"], s=2)
ax[2].scatter(db_o["theoretical_J"], db_o["signal_width"], s=2)
ax[3].scatter(db_nf["theoretical_J"], db_nf["signal_width"], s=2)
ax[4].scatter(db_ocf["theoretical_J"], db_ocf["signal_width"], s=2)
plt.show()

#%% 4 signal_noise_ratio vs calculated_J
base_axis = plt.subplot(5,1,1)
plt.subplot(5,1,1)
plt.scatter(db_original["signal_noise_ratio"], db_original["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_original["Cluster"]))
plt.subplot(5,1,2, sharex=base_axis)
plt.scatter(db_of["signal_noise_ratio"], db_of["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_of["Cluster"]))
plt.subplot(5,1,3, sharex=base_axis)
plt.scatter(db_o["signal_noise_ratio"], db_o["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_o["Cluster"]))
plt.subplot(5,1,4, sharex=base_axis)
plt.scatter(db_nf["signal_noise_ratio"], db_nf["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_nf["Cluster"]))
plt.subplot(5,1,5, sharex=base_axis)
plt.scatter(db_ocf["signal_noise_ratio"], db_ocf["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_ocf["Cluster"]))
plt.show()

#%% 5 theoretical vs calculated J
plt.subplot(2,3,1).set_title("og")
plt.scatter(db_original["theoretical_J"], db_original["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_original["Cluster"]))
plt.subplot(2,3,2).set_title("old flattened")
plt.scatter(db_of["theoretical_J"], db_of["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_of["Cluster"]))
plt.subplot(2,3,3).set_title("old")
plt.scatter(db_o["theoretical_J"], db_o["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_o["Cluster"]))
plt.subplot(2,3,4).set_title("new flattened")
plt.scatter(db_nf["theoretical_J"], db_nf["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_nf["Cluster"]))
plt.subplot(2,3,5).set_title("old curved flattened")
plt.scatter(db_ocf["theoretical_J"], db_ocf["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], db_ocf["Cluster"]))
plt.show()
#%% 6 ssim
plt.plot(range(len(db_ocf["ssim"])), sorted(db_ocf["ssim"]))
plt.show()
#%% 8 on old-flat compare new & old SNR values
plt.scatter(db_ofp["prev_snr"], db_ofp["signal_noise_ratio"])
plt.show()
# %%
