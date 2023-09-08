import numpy as np
from tqdm import tqdm

import itertools
import json
import math
from multiprocessing import Pool
from os import path
import random
import sys

from signal_simulation.signal_generator import SimulatedSignal, simulate_single_signal

def wrapper(args):
    "Wrap simulate_single_signal call cause there is no istarmap_unordered"
    return simulate_single_signal(*args)

def status(iterable, total_size):
    processed_count = 0
    for i in iterable:
        processed_count += 1
        percentage = math.floor(processed_count/total_size * 100)
        print(f"\r{processed_count} out of {total_size}\t{percentage}%", end="", flush=True)
        yield i

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

def generate_database(name: str):
    CURRENT_DIR = path.dirname(__file__)
    OUTPUT_DB_FILE = path.join(CURRENT_DIR, f"../assets/{name}.json")

    signal_width_values = np.arange(0.5, 24.0, 0.5)
    theoretical_J_values = np.arange(0.5, 12.0, 0.05)
    noise_scale_values = adjusted_noise_scale_parameters(10)

    with Pool() as pool:
        parameters = itertools.product(theoretical_J_values, noise_scale_values, signal_width_values)
        parameters = list(parameters)
        parameters = [random.choice(parameters) for _ in range(50)]#Sampler
        print(f"Simulating {len(parameters)} different signals")

        signals_db_asy = pool.imap_unordered(wrapper, parameters)
        signals_db = status(signals_db_asy, len(parameters))# Progress bar
        signals_db = pool.map(vars, signals_db)

    with open(OUTPUT_DB_FILE, "w", encoding="utf-8") as out_file:
        out_file.write(json.dumps(signals_db))

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        sys.exit("Pass the database name as a single argument")
    generate_database(args[1])
