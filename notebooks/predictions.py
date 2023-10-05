#%% imports
import math
from os import path
import sys
from typing import List, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema
import tensorflow as tf

modules_path = path.join(path.dirname(__file__), '../src')
sys.path.append(modules_path)
modules_path = path.join(path.dirname(__file__), '../src/AI_algorithms')
sys.path.append(modules_path)

from baseline_fitting.baselinefit import Signal, baseline_flattening

#%% Reading a slc signal
sig = Signal.leer_archivo("../assets/SeñalesExperimentales/ha_15.slc")

# %%
#MOVE TO ANOTHER FILE LATER
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

def exp_noise (xx, yy, r1, r2):

    aa = np.array(yy)
    ss = np.array(xx) #shape: (1000,)
    subsetter = np.where((ss <= int(r2)) & (ss >= int(r1)))
    #subsetter = np.where((ss <= int(r2 * 1.3)) & (ss >= int(r1 * 0.3)))
    

    only_noise = np.delete(aa, subsetter)
    for i in range (len(only_noise)):
        if only_noise[i] < 0:
            only_noise[i] = 0
        
        else :
            only_noise [i]=i
    #print('solo ruido: ',only_noise)
    return only_noise #return an array with only signal noise

def det_width (xx, yy):
    # create a spline of the multiplet
    spline = UnivariateSpline(xx, yy-np.max(yy)/2, s=0)
    raices = spline.roots() # find the roots

    r1 = raices[0]
    r2 =raices[-1]    
    w = r2-r1
    return w, r1, r2

def calculate_noise(original:List[float], noise:List[float]) -> Tuple[float, float, float]:
    "Calculate signal to noise ratio and related values"
    prom = sum(noise)/len(noise)
    msex = list(map(lambda x: np.square(x - prom), noise))
    mse = sum(msex)/(len(noise) - 1)
    rmse = pow(mse, 0.5) #equivalent to standard deviation
    maxim = max(original)# nos da el maximo de la simulacion original
    s_n = float('inf') if rmse == 0 else maxim / rmse#relacion señal ruido
    return mse, rmse, s_n

def harmonics(expected_first_harmonic:float, minimums:List[float]) -> float:
    "determine which minimum fits the first harmonic"
    
    candidates = [expected_first_harmonic + x for x in range(-2,3)]
    for c in candidates:
        for m in minimums:
            if math.isclose(c,m):
                return c
    return None

#le flatt
import matplotlib.pyplot as plt
plt.plot(sig.x_values, sig.y_values, color="#ff0000")
#baseline_flattening(sig, polynomials=1)
plt.plot(sig.x_values, sig.y_values, color="#00ff00")
plt.show()

w, r1, r2 = det_width(sig.x_values, sig.y_values)
noise_points = exp_noise(sig.x_values, sig.y_values, r1, r2)
mse, rmse, s_n = calculate_noise(sig.y_values, noise_points)

intervalo = len(sig.y_values) // 4
integrs = jdoubling(sig.y_values, intervalo, 164)
plt.plot(range(0,len(integrs)), integrs)
plt.show()

busqueda = int(intervalo/8)
minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo
minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido
armonic1 = int((minimosR[-1])/3)  
subarmos= harmonics(armonic1, minimos)

Jota = minimosR[-1] * sig.gap
distance = Jota - (subarmos * sig.gap)
#Jota, s_n

#Prints
print("J determinada: ", Jota)
print("1 subhrmonico: ", armonic1 * sig.gap)
print("distancia: ", distance)
print("Ancho de señal: ", w)
print("Relación señal/ruido: ", s_n)

# %%
from AI_algorithms.neural_net_training import JdoublingErrorPredictor
from AI_algorithms.gaussian_process_regression import JdoublingResultPredictor
from AI_algorithms.svm import JdoublingReliabilityPredictor

result_predictor = JdoublingResultPredictor()
predicted_j = result_predictor.predict(w, subarmos*sig.gap, distance, s_n)
predicted_j = tf.get_static_value(predicted_j)
print(f"GPR prediction: Predicts a J value of {predicted_j}")

error_predictor = JdoublingErrorPredictor()
predicted_error = error_predictor.predict(Jota, w, subarmos*sig.gap, distance, s_n).numpy()
print(f"Neural net prediction: Predicts Jdoubling error of {predicted_error}")

reliability_predictor = JdoublingReliabilityPredictor()
predicted_reliability = reliability_predictor.predict(s_n, Jota)
print(f"SVM predicted reliability: {predicted_reliability}")

# %%
