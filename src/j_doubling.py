"""The J Doubling algorithm"""

import math

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt

def convolucion(y_vals, n=40, delta_count=128):
    "Convolution between the delta functions and the signal"
    result = np.zeros(len(y_vals) + n*delta_count)
    for i in range(delta_count):
        if i < delta_count //2:
            qq = (-1)**i*y_vals
        else:
            qq = (-1)**(i+1)*y_vals

        for j,y in enumerate(qq):
            result[n*i +j] += qq[j]
    return result

kow_tools = {}

def jdoubling(yy, intervalo:int = 60, m:int = 8):
    "The jdoubling process itself"
    
    integrs = np.zeros(intervalo)
    for i in range(1, intervalo):
        y_new = convolucion(yy, i, m)
        kow_tools[str(i)] = y_new
        integral = sum(abs(y_new))
        integrs[i] = integral
    return integrs

def det_width (xx, yy):
    # create a spline of the multiplet
    spline = UnivariateSpline(xx, yy-np.max(yy)/2, s=0)
    raices = spline.roots() # find the roots

    r1 = raices[0]
    r2 =raices[-1]    
    w = r2-r1
    return w, r1, r2

def calculate_noise(y_values) -> float:
    "Calculate signal to noise ratio using technique from DESPERATE"
    return np.max(y_values)/np.std(y_values)

def harmonics(expected_first_harmonic:float, minimums) -> float:
    "determine which minimum fits the first harmonic"
    
    candidates = [expected_first_harmonic + x for x in range(-2,3)]
    for c in candidates:
        for m in minimums:
            if math.isclose(c,m):
                return c
    return None # CHECK UP ON VALIDITY OF AI 

def jdoubling_process(sig):
    intervalo = len(sig.y_values) // 4
    integrs = jdoubling(sig.y_values, intervalo, 164)
    minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido

    return (integrs, minimos)

def calculate_jdoubling_properties(chosen_candidate, minimums, sig):
    armonic1 = int((chosen_candidate)/3)  
    subarmos= harmonics(armonic1, minimums)

    Jota = chosen_candidate * sig.gap
    if subarmos is None:
        distance = None
        subharmonic = None
    else:
        distance = Jota - (subarmos * sig.gap)    
        subharmonic = subarmos * sig.gap

    w, r1, r2 = det_width(sig.x_values, sig.y_values)
    s_n = calculate_noise(sig.y_values)

    return {
        "calculated_J": Jota,
        "hertz_distance": distance,
        "signal_width": w,
        "signal_noise_ratio": s_n,
        "has_harmonic": False if subarmos == None else True,
        "sub_harmonic": subharmonic
    }

def jdoubling_default_choice(sig):
    (integrals, minimums) = jdoubling_process(sig)
    plt.plot(range(len(integrals)), integrals)
    print(minimums)
    return calculate_jdoubling_properties(minimums[-1], minimums, sig)