"""Fit a signal to a flat horizontal line. Useful for pre-processing

The algorithm implemented here is an adapted version of:
"A General Baseline-Recognition and Baseline-Flattening Algorithm", Pearson A, 1976
"""
#NOTE: check out https://stackoverflow.com/questions/29156532/python-baseline-correction-library

import signal.Signal
import math
import numpy as np
from numpy.polynomial import Polynomial

# Constants used by the algorithm to control the "agresiveness" of the fitting.
# We keep to the values proposed in the paper.
U_PARAMETER = 4.0
# Must be within 2.0 and 3.0
V_PARAMETER = 2.5
CORRECTION_LIMIT_MULTIPLIER = 0.125

def heaviside(x):
    "Step function at 0, with 0 as output in the discontinuity"
    return np.heaviside(x, 0.0)

def calculate_sigma(y_values, previous_sigma):
    """Calculates a measure similar to an standard deviation"""
    ## sqrt((sum(Y(xi)**2 * heaviside(u*sig1 - |Y(xi)|)) / (1 + sum(heaviside(u*sig1 - |Y(xi)|)))
    numerator = np.sum([y**2 * heaviside(U_PARAMETER * previous_sigma - abs(y)) for y in y_values])
    denominator = 1 + np.sum([heaviside(U_PARAMETER * previous_sigma - abs(y)) for y in y_values])
    return np.sqrt(numerator / denominator)

#def calculate_first_sigma(y_values):
#    """Calculates the sigma for the case where the previous sigma would have been infinity"""
#    numerator = np.sum([y**2 for y in y_values])
#    denominator = 1 + np.sum([1 for y in y_values])
#    return np.sqrt(numerator / denominator)

def filter_within_bounds(values, lower, upper):
    """Returns a list where all the values are within the upper and lower bound"""
    return list( filter( lambda x: x > lower and x < upper, values))

def baseline_flattening(source: Signal):
    """Flatten a given signal

    Denoises/Normalizes a given signal
    """

    is_correction_negligible = False
    while not is_correction_negligible:
        #sigma convergence loop
        old_sigma = 0.0
        next_sigma = float("inf")
        while not math.isclose(old_sigma, next_sigma):
            old_sigma = next_sigma
            y_values_within = filter_within_bounds(source.y_values, -U_PARAMETER*old_sigma, U_PARAMETER*old_sigma)
            next_sigma = calculate_sigma(y_values_within, old_sigma)
        # fitting curve
        points_within_v = zip(source.x_values, source.y_values)
        points_within_v = filter(lambda p: p[1] > -V_PARAMETER*old_sigma and p[1] < V_PARAMETER*old_sigma, points_within_v) 
        fitted_polynomial = Polynomial.fit([p[0] for p in points_within_v], [p[1] for p in points_within_v], 8)
        # Substracting fitted curve to original signal
        source.y_values = list(map(lambda i: source.y_values[i] - fitted_polynomial(source.x_values[i])), range(0, len(source.y_values)))
        # Deciding if the correction was negligible
        # |g(x)| < 0.125 * next_sigma for all x
        is_correction_negligible = True
        for x in source.x_values:
            is_correction_negligible = is_correction_negligible and abs(fitted_polynomial(x)) < 0.125 * next_sigma
