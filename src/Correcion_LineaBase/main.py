#!/bin/env python

# Entry point for the program

#%%
from nmrsignal import Signal, gen_random_signal
from baselinefit import baseline_flattening

import matplotlib.pyplot as plt

signal = gen_random_signal(1000, 0, 1000, -50, 50)
plt.plot(signal.x_values, signal.y_values, color="gray")
baseline_flattening(signal)
plt.plot(signal.x_values, signal.y_values, color="green")
plt.plot([signal.x_values[0], signal.x_values[-1]], [signal.y_values[0], signal.y_values[-1]], color="red")
plt.show()
