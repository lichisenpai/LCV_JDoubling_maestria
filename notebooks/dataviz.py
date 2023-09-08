from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT_DIR = path.dirname(__file__)
NEW = path.join(CURRENT_DIR, "../assets/new_noise-db.json")
OLD = path.join(CURRENT_DIR, "../assets/old_noise-db.json")
ORIGINAL = path.join(CURRENT_DIR, "../assets/original-db.json")

dn = pd.read_json(NEW)
do = pd.read_json(OLD)
dinipro = pd.read_json(ORIGINAL)

#0 some stuff
def max_aceptado (x):
    return x * 0.95

def min_aceptado (x):
    return x * 1.05

do['Max_acept'] = do['theoretical_J'].apply(max_aceptado)
do['Min_acept'] = do['theoretical_J'].apply(min_aceptado)
dn['Max_acept'] = dn['theoretical_J'].apply(max_aceptado)
dn['Min_acept'] = dn['theoretical_J'].apply(min_aceptado)
dinipro['Max_acept'] = dinipro['Jref'].apply(max_aceptado)
dinipro['Min_acept'] = dinipro['Jref'].apply(min_aceptado)

cond_do = [(do.calculated_J >= do.Max_acept) & (do.calculated_J <= do.Min_acept), 
                   (do.calculated_J < do.Max_acept) & (do.calculated_J > do.Min_acept)]
cond_dn = [(dn.calculated_J >= dn.Max_acept) & (dn.calculated_J <= dn.Min_acept), 
                   (dn.calculated_J < dn.Max_acept) & (dn.calculated_J > dn.Min_acept)]
cond_dinipro = [(dinipro.Jdet >= dinipro.Max_acept) & (dinipro.Jdet <= dinipro.Min_acept), 
                   (dinipro.Jdet < dinipro.Max_acept) & (dinipro.Jdet > dinipro.Min_acept)]

elecciones = np.array((0, 1), dtype="int8")
Y_do = do['Cluster'] = np.select(cond_do, elecciones, -1)
elecciones = np.array((0, 1), dtype="int8")
Y_dn = dn['Cluster'] = np.select(cond_dn, elecciones, -1)
elecciones = np.array((0, 1), dtype="int8")
Y_dinipro = dinipro['Cluster'] = np.select(cond_dinipro, elecciones, -1)


#1 plotting set parameters
#f, ax = plt.subplots(3)
#ax[0].scatter(do["theoretical_J"], do["signal_width"], s=5)
#ax[1].scatter(dn["theoretical_J"], dn["signal_width"], s=5)
#ax[2].scatter(dinipro["Jref"], dinipro["Width"], s=5)
#plt.show()


#2 plotting signal_noise_ratio:seeing coverage
#f, ax = plt.subplots(3)
#ax[0].scatter(range(len(do["signal_noise_ratio"])), sorted(do["signal_noise_ratio"]), s=2)
#ax[0].set_title('old')
#ax[1].scatter(range(len(dn["signal_noise_ratio"])), sorted(dn["signal_noise_ratio"]),s=2)
#ax[1].set_title('new')
#ax[2].scatter(range(len(dinipro["S/n"])), sorted(dinipro["S/n"]),s=2)
#ax[2].set_title('original')
#plt.show()


#3 theoretical vs calculated J
#f, ax = plt.subplots(3)
#ax[0].scatter(do["theoretical_J"], do["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_do))
#ax[0].set_title('old')
#
#ax[1].scatter(dn["theoretical_J"], dn["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dn))
#ax[1].set_title('new')
#
#ax[2].scatter(dinipro["Jref"], dinipro["Jdet"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dinipro))
#ax[2].set_title('original')
#plt.show()


#4 signal_noise_ratio vs calculated_J
base_axis= plt.subplot(3,1,1)
plt.scatter(do["signal_noise_ratio"], do["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_do))
plt.subplot(3,1,2, sharex=base_axis)
plt.scatter(dn["signal_noise_ratio"], dn["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dn))
plt.subplot(3,1,3, sharex=base_axis)
plt.scatter(dinipro["S/n"], dinipro["Jdet"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dinipro))
plt.show()

#fig = plt.figure(1)
#ax = [1,1,1]
#ax[0] = fig.add_subplot()
#ax[0].scatter(do["signal_noise_ratio"], do["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_do))
#ax[0].set_title('old')
#
#ax[1] = fig.add_subplot(sharex=ax[0])
#ax[1].scatter(dn["signal_noise_ratio"], dn["calculated_J"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dn))
#ax[1].set_title('new')
#
#ax[2] = fig.add_subplot(sharex=ax[0])
#ax[2].scatter(dinipro["S/n"], dinipro["Jdet"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_dinipro))
#ax[2].set_title('original')
#plt.show()


#5 signal_noise vs calculated_J vs width for old
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(do["signal_noise_ratio"], do["calculated_J"], do["signal_width"], s=2, c=np.take(['#00ff00', '#ff0000'], Y_do))
#
#plt.show()

#6 theoretical j, calculated j (with choices)