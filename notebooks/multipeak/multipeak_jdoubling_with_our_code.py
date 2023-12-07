#%% 
from os import path
import sys

import matplotlib.pyplot as plt

modules_path = path.join(path.dirname(__file__), '../../src')
sys.path.append(modules_path)
from baselinefit import baseline_flattening
from nmrsignal import Signal
from j_doubling import convolucion, jdoubling_default_choice, signal_deconvolution

#%%
signal = Signal.read_file_mnova_csv("../../assets/mnova-signals/cpto-1111.csv", 700)
#signal = Signal.read_file_mnova_csv("out1.csv", 700)
plt.plot(signal.x_values, signal.y_values)
plt.show()


#%%
props = jdoubling_default_choice(signal)
plt.plot(props["integral"])
plt.show()

estimated_j = props["calculated_J"]
print(f"Jota: {estimated_j}")
# %%
# in the land of deconvolutions

next_signal = signal_deconvolution(signal, props)
plt.plot(signal.y_values, color="#999999")
plt.plot(next_signal.y_values, color="#ff00ff")
plt.show()

next_signal.write_file_mnova_csv("out1.csv", 700)