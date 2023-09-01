import numpy as np
from tqdm import tqdm

import itertools
import json
import math
from multiprocessing import Pool
from os import path

from signal_simulation.signal_generator import SimulatedSignal, simulate_single_signal

CURRENT_DIR = path.dirname(__file__)
OUTPUT_DB_FILE = path.join(CURRENT_DIR, "../assets/_ai_db.json")

def wrapper(args):
    "Wrap simulate_single_signal call cause there is no istarmap_unordered"
    return simulate_single_signal(*args)

def adjusted_noise_scale_parameters(count):
    """Generates noise_scales that produce good signal_noise ratios after simulation.

    The closer the noise_scales are to 0 the better their corresponding ratios.
    We need a full fledged function because the relationship between scales and ratios
    seem to behave exponentially (very small values covering most of the ratio space)
    """
    bases = list(range(0,count))
    max = bases[-1]
    formula = lambda z: math.exp(z)/math.exp(max)
    noises = map(formula, bases)
    return list(noises)

signal_width_values = np.arange(0.5, 24.0, 0.5)
theoretical_J_values = np.arange(0.5, 12.0, 0.05)
noise_scale_values = adjusted_noise_scale_parameters(15)

with Pool() as pool:
    parameters = itertools.product(theoretical_J_values, noise_scale_values, signal_width_values)
    parameters = list(parameters)
    print(f"Simulating {len(parameters)} different signals")

    signals_db_asy = pool.imap_unordered(wrapper, parameters)
    # Progress bar
    signals_db = tqdm(signals_db_asy, total=len(parameters))
    signals_db = pool.map(vars, signals_db)

with open(OUTPUT_DB_FILE, "w", encoding="utf-8") as out_file:
    out_file.write(json.dumps(signals_db))
