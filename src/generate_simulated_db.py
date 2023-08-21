import numpy as np

import itertools
import json
from multiprocessing import Pool
from os import path

from signal_simulation.signal_generator import SimulatedSignal, simulate_single_signal

CURRENT_DIR = path.dirname(__file__)
OUTPUT_DB_FILE = path.join(CURRENT_DIR, "../assets/ai_db.json")

theoretical_J_values = np.arange(0.5, 12.0, 0.05)
noise_scale_values = np.arange(0.01, 1, 0.01)
signal_width_values = np.arange(0.5, 24.0, 0.5)

with Pool() as pool:
    parameter_iterator = itertools.product(theoretical_J_values, noise_scale_values, signal_width_values)
    signals_db = pool.starmap(simulate_single_signal, parameter_iterator)
    signals_db = pool.map(vars, signals_db)

with open(OUTPUT_DB_FILE, "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(signals_db))