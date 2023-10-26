#%% 1
# A first idea is extending the curve into a whole signal and then baseline fitting
# recutting at the end

import math
from os import path
import sys
import itertools

import matplotlib.pyplot as plt

modules_path = path.join(path.dirname(__file__), '../src')
sys.path.append(modules_path)
from baseline_fitting.baselinefit import baseline_flattening
from signal_simulation.signal_generator import *
from baseline_fitting.nmrsignal import Signal

signal = Signal.leer_archivo("../assets/SeñalesExperimentales/ha_04.slc")

# which signals and why?
# 09 -> ideal curve, should not be modified much
#   GOOD PARAMETERS: poly=2, flat_rad=2
# 04 -> a bad curve


# 25

def retool(signal, flattened_rad=1):
    assert flattened_rad >= 0 and isinstance(flattened_rad, int)
    sig_len = len(signal.y_values)
    flat_extension = list(itertools.repeat(signal.y_values[0], times=flattened_rad*sig_len))
    flat_extension_r = list(itertools.repeat(signal.y_values[-1], times=flattened_rad*sig_len))
    extended_y = [*flat_extension, *signal.y_values, *flat_extension_r]
    #extended_y = flat_extension + signal.y_values + flat_extension
    extended_min_hertz = signal.min_hertz - signal.gap*sig_len*flattened_rad
    extended_max_hertz = signal.max_hertz + signal.gap*sig_len*flattened_rad
    return Signal(extended_y, extended_min_hertz, extended_max_hertz)

def untool(signal, flat_rad=1):
    og_len = signal.point_count // (flat_rad*2+1)
    reduced_y = signal.y_values[og_len*flat_rad:og_len*flat_rad + og_len]
    reduced_min_hertz = signal.min_hertz + signal.gap*og_len*flat_rad
    reduced_max_hertz = signal.max_hertz - signal.gap*og_len*flat_rad
    return Signal(reduced_y, reduced_min_hertz, reduced_max_hertz)


plt.plot(signal.x_values, signal.y_values, color='#ff0000')
plt.show()

extended = retool(signal, 1)

plt.plot(extended.x_values, extended.y_values, color='#ff00ff')
plt.plot(signal.x_values, signal.y_values, color='#ff0000')
plt.show()

plt.plot(extended.x_values, extended.y_values, color='#ff00ff')
baseline_flattening(extended, 2)
plt.plot(extended.x_values, extended.y_values, color='#00ffff')
plt.show()

reduced = untool(extended, 1)

plt.plot(signal.x_values, signal.y_values, color='#aaaaaa')
plt.plot(reduced.x_values, reduced.y_values, color='#00ff00')
#baseline_flattening(signal, 1)
#plt.plot(signal.x_values, signal.y_values, color='#ff0000')
plt.show()

def signal_slope(sig):
    return (sig.y_values[0] - sig.y_values[-1])/(sig.x_values[0] - sig.x_values[-1])

print(f"og slope {signal_slope(signal)}")
print(f"new slope {signal_slope(reduced)}")

# y = mx+b
# m = (y1-y2)/(x1-x2)

#%% all at once
signal_paths = [
    "../assets/SeñalesExperimentales/ha_04.slc",
    "../assets/SeñalesExperimentales/ha_07.slc",
    "../assets/SeñalesExperimentales/ha_09.slc",
    "../assets/SeñalesExperimentales/ha_12.slc",
    "../assets/SeñalesExperimentales/ha_15.slc",
    "../assets/SeñalesExperimentales/ha_19.slc",
    "../assets/SeñalesExperimentales/ha_22.slc",
    "../assets/SeñalesExperimentales/ha_24.slc",
    "../assets/SeñalesExperimentales/ha_06.slc",
    "../assets/SeñalesExperimentales/ha_08.slc",
    "../assets/SeñalesExperimentales/ha_10.slc",
    "../assets/SeñalesExperimentales/ha_13.slc",
    "../assets/SeñalesExperimentales/ha_17.slc",
    "../assets/SeñalesExperimentales/ha_21.slc",
    "../assets/SeñalesExperimentales/ha_23.slc",
    "../assets/SeñalesExperimentales/ha_25.slc"
]

def evaluate(signal_paths, poly_deg, flat_rad):
    for s in signal_paths:
        s = Signal.leer_archivo(s)
    
        moded = retool(s, flat_rad)
        baseline_flattening(s, poly_deg)
        moded = untool(moded, flat_rad)

        base_slope = signal_slope(s)
        res_slope = signal_slope(moded)
        yield abs(0.0 - res_slope)

def evaluate_parameters(poly_deg, flat_rad):
    errors = evaluate(signal_paths, poly_deg, flat_rad)
    errors = list(errors)
    return sum(errors) / len(errors)

min_err = None
for pd in range(1,10):
    for fr in range(9):
        avg_err = evaluate_parameters(pd, fr)
        print(f"degree:{pd}, flat radius:{fr} -> avg error:{avg_err}")
        if min_err is None or avg_err < min_err[0]:
            min_err = (avg_err, pd, fr)
print(f"Best parameters found >> deg:{min_err[1]} frad:{min_err[2]} with error {min_err[0]}")

#%%

best_deg = min_err[1]
best_frad = min_err[2]

def giga_flat(sig, deg,frad):
    sig = retool(sig, frad)
    baseline_flattening(sig, deg)
    return untool(sig ,frad)


for s in signal_paths:
    s = Signal.leer_archivo(s)
    o = giga_flat(s, 4, 2)
    p = giga_flat(s, 1, 2)
    plt.plot(s.x_values, s.y_values, color='#aaaaaa')
    plt.plot(o.x_values, o.y_values, color='#ff0000')
    plt.plot(p.x_values, p.y_values, color='#0000ff')
    plt.show()

# BEST DETERMINED PARAMETERS: frad:2, deg:4