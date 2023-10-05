"""Simulation of NMR signals for ML algorithms"""

import math
import random
from typing import List, Tuple

import numpy as np
from numpy.polynomial import Polynomial
from nmrsim import Multiplet
from nmrsim.math import add_lorentzians
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt

from baseline_fitting.nmrsignal import Signal, gen_random_signal
from baseline_fitting.baselinefit import baseline_flattening


class SimulatedSignal:
    """All of the data used or produced by simulating a signal using nmrsim.

    The goal of this class is to act as a single instance for the Neural net training.
    """

    def __init__(self, theoretical_J: float, calculated_J: float, JDoubling_error: float, has_harmonic:bool, sub_harmonic: float, hertz_distance: float, signal_noise_ratio: float, signal_width: float, ssim: float):
        self.theoretical_J = theoretical_J
        "J value passed as input into the simulation"
        self.calculated_J = calculated_J
        "J value calculated by the JDoubling algorithm (First minimum on integral graph)"
        self.JDoubling_error = JDoubling_error
        "Difference between the calculated J value and the J input into simulation"
        self.JDoubling_quadratic_error = JDoubling_error ** 2
        "Squared JDoubling error"
        self.has_harmonic = has_harmonic
        "Whether a sub_harmonic could be found. If false then the value of sub_harmonic is meaningless"
        self.sub_harmonic = sub_harmonic
        "Second minimum on the integral graph"
        self.hertz_distance = hertz_distance
        "Distance between the first(J) and second(sub_harmonic) minimums on the integral graph. In hertz"
        self.signal_noise_ratio =  signal_noise_ratio
        "Signal to noise ratio (Passed as input into simulation)"
        self.signal_width = signal_width
        "Width of the signal in hertz"
        self.ssim = ssim
        "Structural similarity index, measures how similar the overall signal is after noising and flattening"
    
    def __str__(self) -> str:
        return f"""
theoretical_J:{self.theoretical_J},
calculated_J:{self.calculated_J},
err:{self.JDoubling_error},
q_err:{self.JDoubling_quadratic_error},
has_harmonic:{self.has_harmonic},
sub harmonic:{self.sub_harmonic},
hertz distance:{self.hertz_distance},
signal to noise:{self.signal_noise_ratio},
signal width:{self.signal_width},
ssim:{self.ssim}
"""

    def __repr__(self) -> str:
        return str(self)
    
    @staticmethod
    def from_dict(source):
        """Build Simulated Signal from dict generated with vars()
        
        Will throw errors with invalid dicts
        """
        return SimulatedSignal(
            source["theoretical_J"],
            source["calculated_J"],
            source["Jdoubling_error"],
            source["has_harmonic"],
            source["sub_harmonic"],
            source["hertz_distance"],
            source["signal_noise_ratio"],
            source["signal_width"],
            source["ssim"])


def simulate_single_signal(theoretical_J: float, noise_scale: float, signal_width:float ) -> SimulatedSignal:
    "Simulate a signal given it's theoretical J value and the noise scaler desired"

    [desplazamiento, intensidades] = multiplet(1200.0, 1, theoretical_J, 1, signal_width)
    base_signal = Signal(intensidades, desplazamiento[0], desplazamiento[-1])

    #Simulating instrumental noise and subsequent flattening
    signal = simulate_old_noise(base_signal, noise_scale)
    signal = simulate_non_flat_defects(signal, noise_scale)
    plt.plot(signal.x_values, signal.y_values, color='#cccccc')
    baseline_flattening(signal)
    plt.plot(signal.x_values, signal.y_values, color='#ffaa00')
    _mse, _rmse, signal_noise_ratio = calculate_noise(base_signal.y_values, signal.y_values - base_signal.y_values)

    ssim = structural_similarity_index(signal, base_signal)

    # 1.3 is meant to reduce the amount of operations during integration.
    intervalo = int((theoretical_J / signal.gap) * 1.3)
    m = 164 # number of delta functions used

    integrs = jdoubling(signal.y_values, intervalo, m)
    # We found that 7 works best for finding the right minimum values on the integral graph
    busqueda = int(intervalo/7)
    minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo
    minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido
    calculated_J = minimosR[-1] * signal.gap

    # Searching for the first subharmonic value, we know that it must be around J/3
    armonic1 = int((minimosR[-1])/3)
    subarmos= harmonics(armonic1, minimos)
    if subarmos == None:
        # Could not find first subharmonic
        subarmosHz = 0
        has_harmonic = False
    else:
        subarmosHz = subarmos * signal.gap
        has_harmonic = True

    return SimulatedSignal(theoretical_J, calculated_J, theoretical_J - calculated_J, has_harmonic, subarmosHz, calculated_J - subarmosHz, signal_noise_ratio, signal_width, ssim)

# The amount of sample points that we take from the simulated signal
RESOLUTION_POINTS = 1000

def multiplet(v:float, I:int, J:float, r:int, signal_width:float):
    """The signal simulation itself

    v -> chemical shift, Hz
    I -> Signal integration, number of hydrogens
    J -> coupling constant
    r -> number of h neighbours
    """

    min_x = v - 20 
    max_x = v + 20
    td = Multiplet(v , I, [(J, r)])

    # BIG change below, from reading the source of nmrsim it seems that the commented line
    # is equivalent to the block below, but without using matplotlib at all
    # (which makes sense since we shouldn't need a plotter for simulation)
    # grafica = mplplot(td.peaklist(), points=RESOLUTION_POINTS, w=signal_width, limits=(min_x, max_x), hidden=True)

    # MODIFIED FROM nmrsim's SOURCE
    peaklist = td.peaklist()
    peaklist.sort()
    x = np.linspace(min_x, max_x, RESOLUTION_POINTS)
    y = add_lorentzians(x, peaklist, signal_width)
    return x, y

def calculate_noise(original:List[float], noise:List[float]) -> Tuple[float, float, float]:
    "Calculate signal to noise ratio and related values"
    prom = sum(noise)/len(noise)
    msex = list(map(lambda x: np.square(x - prom), noise))
    mse = sum(msex)/(len(noise) - 1)
    rmse = pow(mse, 0.5) #equivalent to standard deviation
    maxim = max(original)# nos da el maximo de la simulacion original
    s_n = float('inf') if rmse == 0 else maxim / rmse#relacion señal ruido
    return mse, rmse, s_n

def aplicar(x:List[float], y:List[float], start_index:int) -> List[float]:
    """Add the values from list y to list x starting at the position start_index. Return a new list.
    
    len(x) must be the len(y) + start_index
    """
    #assert(len(x) == len(y) + start_index)

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

def harmonics(expected_first_harmonic:float, minimums:List[float]) -> float:
    "determine which minimum fits the first harmonic"
    
    candidates = [expected_first_harmonic + x for x in range(-2,3)]
    for c in candidates:
        for m in minimums:
            if math.isclose(c,m):
                return c
    return None

def simulate_old_noise(signal: Signal, noise_scale:float) -> Signal:
    new_y_vals = []
    for _ in signal.x_values:
        new_y_vals.append(random.random() / noise_scale)

    new_y = np.add(signal.y_values, new_y_vals)
    return Signal(new_y, signal.min_hertz, signal.max_hertz)

def simulate_non_flat_defects(signal: Signal, noise_scale: float) -> Signal:
    """We simulate the kinds of shapes NMR signals can take in non-ideal conditions

    This function combined with simulate_old_noise should turn a perfect theoretial
    signal into one that could have been plausibly produced by a real instrument.
    """
    try:
        degree = random.randrange(5)+1
        max_height = max(signal.y_values)
        points_x = random.choices(signal.x_values, k=degree)
        points_y = [random.random()*max_height for x in points_x]
        curve = Polynomial.fit(points_x, points_y, degree+2)
        new_y_values = []
        for i in range(len(signal.x_values)):
            new_y_values.append(curve(signal.x_values[i]) + signal.y_values[i])
        return Signal(new_y_values, signal.min_hertz, signal.max_hertz)
    except Exception as e:
        # The polynomial fit can fail
        print("failed fitting")
        return signal

def structural_similarity_index(measured_signal: Signal, simulated_signal: Signal) -> float:
    """Gives a measure of the uniformity of the spectrum as compared to a simulated measurement
    
    Described in a newer baseline-fitting paper by Gerald A. Pearson. Who also designed the
    fitting algorithm we use.
    A value of SSIM=1 indicates that the two signals are identical
    """

    #Need
    # X = real measurement
    # Y = simulated
    # mean of a signal
    # std-dev of a signal
    # (2mxmy)(2sXY) / (mx**2  + my**2)(sX**2 + sY**2) - c
    # 2\sigma_{XY} means -> 2*np.cov(X,Y)[0][1]
    # c = SSIM(Y,Y) with c=0
    mean_x = np.mean(measured_signal.y_values)
    stddev_x = np.std(measured_signal.y_values)
    mean_y = np.mean(simulated_signal.y_values)
    stddev_y = np.std(simulated_signal.y_values)
    covariance = np.cov(measured_signal.y_values, simulated_signal.y_values)[0][1]

    return (2 * mean_x * mean_y * 2 * covariance) / ((mean_x**2 + mean_y**2)*(stddev_x**2 + stddev_y**2))
