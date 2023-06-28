
class SimulatedSignal:
    """All of the data used or produced by simulating a signal using nmrsim.

    The goal of this class is to act as a single instance for the Neural net training.
    """

    def __init__(self, calculated_J: float, JDoubling_error: float, sub_harmonic: float, hertz_distance: float, signal_noise_ratio: float, signal_width: float):
        self.calculated_J = calculated_J
        "J value calculated by the JDoubling algorithm (First minimum on integral graph)"
        self.Joubling_error = JDoubling_error
        "Difference between the calculated J value and the J input into simulation"
        self.JDoubling_quadratic_error = JDoubling_error ** 2
        "Squared JDoubling error"
        self.sub_harmonic = sub_harmonic
        "Second minimum on the integral graph"
        self.hertz_distance = hertz_distance
        "Distance between the first(J) and second(sub_harmonic) minimums on the integral graph. In hertz"
        self.signal_noise_ratio =  signal_noise_ratio
        "Signal to noise ratio (Passed as input into simulation)"
        self.signal_width = signal_width
        "Width of the signal in hertz"


## TODO: generate an array of Simulated signals and store them into a json

##############################################################################

from Correcion_LineaBase.nmrsignal import Signal

import numpy as np
from nmrsim import Multiplet
from nmrsim.plt import mplplot
import pandas as pd
from scipy.signal import argrelextrema
import json 

def simulate_single_signal(theoretical_J: float, noise_level: float) -> SimulatedSignal:
    ruido = ReadJsonNoise("RandomNoise.json", noise_level)
    [desplazamiento, intensidades] = multiplet(1200.0, 1, theoretical_J, 1)
    señalCruido = ruido + intensidades
    _mse, _rmse, signal_noise_ratio = Noise(intensidades, ruido)

    #TODO: make sure desplazamiento comes out sorted st [0] is the smallest & [-1] the biggest
    signal = Signal(intensidades, desplazamiento[0], desplazamiento[-1])

    #TODO: What are 1.3 and 164 ?
    intervalo = int((theoretical_J / signal.gap) * 1.3)
    m = 164
    integrs = integrar(signal.y_values, intervalo, m)

    #TODO: What does the 7 mean?
    busqueda = int(intervalo/7)
    minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo
    minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido

    #TODO: Again, what does the 3 mean?
    armonic1 = int((minimosR[-1])/3)
    subarmos= Armonics(armonic1, minimos)
    subarmosHz = subarmos * signal.gap

    #WHERE THE J doubling at tho ?

    calculated_J = minimosR[-1] * signal.gap

    return SimulatedSignal(calculated_J, theoretical_J - calculated_J, subarmosHz, calculated_J - subarmosHz, signal_noise_ratio, max_hertz - min_hertz)
##############################################################################


ww = 0.5

def ReadJsonNoise (x, c):
    prueba = pd.read_json(x)
    Y = prueba['Random']#para tenerlo como array
    Rvalue = np.random.choice(Y, 1000)#wlige 1000 puntos al azar, y 1000 por que es el valor de puntos que yo meti por default a mi simulacion
    Norm = list(map(lambda w: w / c, Rvalue))#El c es el valor que divide los valores random de ruido, entre mayor el no. se ve menis el ruido
    return Norm

def Noise (a, b):
    original = a
    noise = b
    prom = sum(noise)/len(noise)
    msex = list(map(lambda x: np.square(x - prom), noise))
    mse = sum(msex)/(len(noise) - 1)
    rmse = pow(mse, 0.5) #la raiz cuadrada del anterior
    maxim = max(original)# nos da el maximo de la simulacion original
    s_n = maxim / rmse #relacion señal ruido
    print("El mse es: ", mse)
    print("El rmse es: ", rmse)
    print("El maximo es: ", maxim)
    print("La relación señal ruido es: ", s_n)
    return mse, rmse, s_n

def multiplet (v, I, J, r):
    #La simulacion tan famosa que ya conoces
    min_x = v - 20 
    max_x = v + 20
    td = Multiplet(v , I, [(J, r)]) 
    grafica = mplplot(td.peaklist(), points=1000, w=ww, limits=(min_x, max_x))
    return grafica 

def archiv_txt (X, Y):
    #Datos en Y
    intensidades = Y
    no_datos = len(intensidades)
    intensidades1 = []
    for i in range (len(intensidades)):
        intensidades1.append(round(intensidades[i], 6))

    #Datos en x 
    Hz = X
    min_Hz = (Hz[0])
    max_Hz = (Hz[no_datos - 1])

    oracion1 = f"Number of data points      : {no_datos} \n"
    oracion2 = f"Chemical shift range (ppm) : 1 1\n"
    oracion3 = f"Chemical shift range (Hz)  : {min_Hz} {max_Hz}\n"

    #escritura del archivo para guardar los datos
    f = open('Prueba.slc','w')
    f.write(oracion1)
    f.write(oracion2)
    f.write(oracion3)
    
    for i in range (len(intensidades1)):
        f.write(str(intensidades1[i])+"\n")
    
    f.close()

#Jdoubling
def leer_archivo(nombre):
    #Hay que analizar las primeras 3 renglones por separado.
    #a, b seran los extremos del intervalo en Hz
    
    arch = open(nombre, "r")
    if arch.mode == 'r':
        contenido = arch.read()
        contenido = contenido.split("\n")
        tam = int(contenido[0].split(": ")[1])
        [a, b] = contenido[2].split(": ")[1].split(" ")

        yy = np.zeros(tam)
        for i in range(0, tam):
            yy[i] = (float(contenido[i+3]))
            
    else:
        print("No existe el archivo")
    
    return [yy, float(a), float(b)]

def aplicar(x, y, m):
    # El tama;o de x tiene que ser del tama;o de y mas m
    #x y y son arreglos
    x_new = np.copy(x)
    for i in range(0, len(y)):
        x_new[m+i] += y[i]
    return x_new

def convolucion(yy, n = 40, m = 128): #que n va de 1 hasta 64
    tamano = len(yy)
    ceros = np.zeros(tamano+n*m)

    for i in range(0, m):

        ceros = aplicar(ceros, (-1)**i*yy, n*i)
    return ceros

#Input: yy
def integrar(yy, intervalo = 60, m = 8):
    #intervalo es el maximo valor de las n
    
    integrs = np.zeros(intervalo)
    
    for i in range(1, intervalo):
        y_new = convolucion(yy, i, m)
        integral = sum(abs(y_new))
        integrs[i] = integral
    return integrs

#Generando un vector con la se;al trasladada 1 vez
def trasladar(ys, n):
    # delta = xs[n]
    tam = len(ys) + n
    y_new = np.zeros(tam)
    
    for i in range( 0, tam ):
        if i < len(ys):
            y_new[i] += ys[i]
        if i >= n:
            y_new[i] += -ys[i-n]
    return y_new    

def new_data (d, E, S, D, sn, E2): #Escribir los datos que quiero dentro del diccionario
    #J= cte. de acoplamiento que yo puse(jota), d= cte. de acoplamiento determinada(calc), E=error 
    dato_n = {}
    
    dato_n['Jref'] = 12.0
    dato_n['Jdet'] = d
    dato_n['Width'] = ww 
    dato_n['Error'] = E 
    dato_n['1Subharmonic'] = S
    dato_n['Distance'] = D
    dato_n['S/n'] = sn
    dato_n['Error2']=E2
    return dato_n

#crear el Json 
def escritura_json (x):
    nombre = f"{x}.json"
    with open(nombre, 'w') as archivo: 
        json.dump(Jota_12_0Hz, archivo)
        #print("Archivo exportado con éxito")
    return 

#Confirmacion subarmonicos 
def Armonics (x, integ):
    armonic1 = x
    if armonic1 in integ: 
        print("armonico coincide perfectamente")
        
        return armonic1 
    
    elif armonic1 + 1 in integ:
        print("armonico coincide +1") 
        
        return armonic1 + 1

    elif armonic1 + 2 in integ: 
        print("armonico coincide +2")
        
        return armonic1 + 2
    
    elif armonic1 - 1 in integ: 
        print("armonico coincide -1")
        return armonic1 - 1
    
    elif armonic1 - 2 in integ:
        print("armonico coincide -1") 
        return armonic1 - 2
    
    else:
        print("subarmonico no confirmado")
        return 0




#jotas = np.linspace(0.5, 12.0, 201)#el intervalo de trabajo de las J´s en las que quiero trabajar
jotas = [12.0 for _ in range(37)]
division = np.arange(5, 301, 8) #len: 60
calc = [] #lista para guardar la J que determina JDoubling
Jota_12_0Hz = [] #lista donde guardar los datos para el json
SubHarmonics = [] #lista que guarda el primer valor del subarmónico
S_n = []

for i in range (len(division)):
    #Para simular 
    J = 12.0
    D = division[i]
    ruido = ReadJsonNoise("RandomNoise.json", D) 
    multiplete = multiplet(1200.0, 1, J, 1) 
    desplazamiento = multiplete[0] 
    intensidades = multiplete[1] 
    señalCruido = ruido + intensidades 
    mse, rmse, s_n = Noise(intensidades, ruido) 

    # FIX: write, then inmediately read file ?
    archiv_txt(desplazamiento, señalCruido)
    #JDoubling
    yy, a, b = leer_archivo('Prueba.slc')

    iz = 0
    de = len(yy) - 1

    # Generar la secuencia xx
    paso_hz = abs(a-b)/len(yy)
    xx = [i*paso_hz+min([a, b]) for i in range(0, len(yy))]

    # FIX: is any of these lines necessary ?
    #Redefiniendo el arreglo en y
    yy = yy[iz:de]
    xx = xx[iz:de]
    nuevo_paso_hz = (xx[-1]-xx[0])/len(yy)

    # La escala en X de la siguiente figura está en enteros. Utilizar paso_Hz para convertir a Hz
    intervalo = int((J / paso_hz) * 1.3)
    m = 164
    integrs = integrar(yy, intervalo, m)


    busqueda = int(intervalo/7)
    minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo

    minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido

    #print(f"valores minimos (+ ruido) en: {minimos}")

    
    armonic1 = int((minimosR[-1])/3)

    subarmos= Armonics(armonic1, minimos)
    subarmosHz = subarmos * paso_hz

    Jota = minimosR[-1] * paso_hz

    calc.append(Jota) 
    SubHarmonics.append(subarmosHz)
    S_n.append(s_n)

    #Seleccionar el mínimo deseado para que se determine la J.¶
    #print(f"Jota: {Jota}         Resolución Digital: {paso_hz}")


Error = list(np.array(jotas)-np.array(calc)) 
Error2 = list(map(lambda x: np.square(x), Error))
DistHz = list(np.array(calc)-np.array(SubHarmonics)) #distancia entre J y el primer subarmónico 


for a, b, c, d, e, f in zip(calc, Error, SubHarmonics, DistHz, S_n, Error2):
    new_entry = new_data(a, b, c, d, e, f)
    Jota_12_0Hz.append(new_entry)
escritura_json("J12_0yW0_5ruido")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                