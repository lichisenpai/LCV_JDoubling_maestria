import numpy as np

import itertools
import json
from multiprocessing import Pool

from signal_simulation.signal_generator import SimulatedSignal, simulate_single_signal

theoretical_J_values = np.arange(0.5, 12.0, 0.04)
noise_scale_values = np.arange(0.1, 1, 0.04)
signal_width_values = np.arange(0.5, 24.0, 0.4)

with Pool() as pool:
    parameter_iterator = itertools.product(theoretical_J_values, noise_scale_values, signal_width_values)
    signals_db = pool.starmap(simulate_single_signal, parameter_iterator)
    signals_db = pool.map(vars, signals_db)

print(json.dumps(signals_db))
