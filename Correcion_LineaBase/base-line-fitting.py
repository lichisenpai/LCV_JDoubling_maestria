import numpy as np
import math

class Signal:
    """Represents and NMR signal"""

    def  __init__(self,  y_values, min_hertz_value, max_hertz_value):
        self.y_values = y_values
        self.min_hertz_value = min_hertz_value
        self.max_hertz_value = max_hertz_value
    
    def __str__(self):
        return f"min: {self.min_hertz_value}    max: {self.max_hertz_value}\n{self.y_values}"

    def leer_archivo(nombre):
    #Hay que analizar las primeras 3 renglones por separado.
    #a, b seran los extremos del intervalo en Hz
        
        with open(nombre, "r") as arch:
            contenido = arch.read()
            contenido = contenido.split("\n")
            tam = int(contenido[0].split(": ")[1])
            [a, b] = contenido[2].split(": ")[1].split(" ")

            yy = np.zeros(tam)
            for i in range(0, tam):
                yy[i] = (float(contenido[i+3]))
            
            return Signal(yy, float(a), float(b))
            
            #return [yy, float(a), float(b)]

# ....
U_PARAMETER = 4.0
# must be within 2.0 and 3.0
V_PARAMETER = 2.5
# If the correction meassure was lower than the limit, then the algorithm may end
# |g(x)| < CORRECTION_LIMIT_MULTIPLIER*sigma_1
CORRECTION_LIMIT_MULTIPLIER = 0.125

def heaviside(x):
    return np.heaviside(x, 0.0)

def calculate_sigma(y_values, previous_sigma):
    y_values = y_values
    ## sqrt(    (sum(Y(xi)**2 * heaviside(u*sig1 - |Y(xi)|)) / (1 + sum(heaviside(u*sig1 - |Y(xi)|)))
    numerator = np.sum([y**2 * heaviside(U_PARAMETER * previous_sigma - abs(y)) for y in y_values])
    denominator = 1 + np.sum([heaviside(U_PARAMETER * previous_sigma - abs(y)) for y in y_values])
    return np.sqrt(numerator / denominator)

def baseline_flatening(source_signal):
    """Flatten a given signal

    Denoises/Normalizes a given signal
    """

    old_sigma = 1000000 # Not sure about this
    next_sigma = calculate_sigma(source_signal.y_values, old_sigma)
    while not math.isclose(old_sigma, next_sigma):
        old_sigma = next_sigma
        next_sigma = calculate_sigma(source_signal.y_values, old_sigma)
    
    # Returning sigma for now, from this point on we should be running curve fitting
    return next_sigma

signal = Signal.leer_archivo("../SeñalesExperimentales/ha_04.slc")
print(baseline_flatening(signal))