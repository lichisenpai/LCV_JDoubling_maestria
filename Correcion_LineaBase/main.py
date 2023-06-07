#!/bin/env python

# Entry point for the program

from signal import Signal, baseline_flattening


signal = Signal.leer_archivo("./SeñalesExperimentales/ha_04.slc")
baseline_flattening(signal)
