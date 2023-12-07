#%% 
from os import path
import sys

import matplotlib.pyplot as plt

modules_path = path.join(path.dirname(__file__), '../../src')
sys.path.append(modules_path)
from baselinefit import baseline_flattening
from nmrsignal import Signal
from j_doubling import convolucion, jdoubling_default_choice

#%%
signal = Signal.read_file_mnova_csv("../../assets/mnova-signals/cpto-1111.csv", 700)
#signal = Signal.read_file_mnova_csv("rezz.csv", 700)
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

deconvolution = convolucion(signal.y_values, props["chosen_minimum"])
plt.plot(deconvolution)
plt.show()

# Extracting our section
original_points = len(signal.y_values)//2
half_point = len(deconvolution)//2 - props["chosen_minimum"]//2
start_point = half_point - original_points
end_point = half_point + original_points

extracted_deconvolution = deconvolution[start_point:end_point]*-1

plt.plot(extracted_deconvolution, color="#ff00ff")
plt.plot(signal.y_values, color="#999999")
plt.show()

# Assembling output
out = ""
for i, val in enumerate(extracted_deconvolution):
    out += f"{i*signal.gap/700}\t{val}\t\n"

with open("rezz.csv", "w") as f:
    f.write(out)