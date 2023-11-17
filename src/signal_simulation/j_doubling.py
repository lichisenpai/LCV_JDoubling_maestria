"""The J Doubling algorithm"""

import math
from typing import List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema


def aplicar(x:List[float], y:List[float], start_index:int) -> List[float]:
    """Add the values from list y to list x starting at the position start_index. Return a new list.
    
    len(x) must be the len(y) + start_index
    """

    result = np.copy(x)
    for i in range(0, len(y)):
        result[start_index + i] += y[i]
    return result

def convolucion(yy:List[float], n:int = 40, m:int = 128) -> List[float]:
    "convolution :)"
    tamano = len(yy)
    ceros = np.zeros(tamano+n*m)

    for i in range(0, m):

        ceros = aplicar(ceros, (-1)**i*yy, n*i)
    return ceros

def jdoubling(yy:List[float], intervalo:int = 60, m:int = 8):
    "The jdoubling process itself"
    
    integrs = np.zeros(intervalo)
    for i in range(1, intervalo):
        y_new = convolucion(yy, i, m)
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

def calculate_noise(y_values:List[float]) -> float:
    "Calculate signal to noise ratio using technique from DESPERATE"
    return np.max(y_values)/np.std(y_values)

def harmonics(expected_first_harmonic:float, minimums:List[float]) -> float:
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
