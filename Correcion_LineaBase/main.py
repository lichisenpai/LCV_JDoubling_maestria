#!/bin/env python

# Entry point for the program

from signal import Signal, gen_random_signal
from baselinefit import baseline_flattening

signal = gen_random_signal(10, 0, 1000, -50, 50)
#print(signal)
baseline_flattening(signal)
print(signal)
