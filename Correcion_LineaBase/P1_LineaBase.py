from nmrsim import Multiplet
#from nmrsim.plt import mplplot
import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab as pl
from statistics import mean
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

print("HERE")

def multiplet (v, I, J, r):
    #La simulacion tan famosa que ya conoces
    min_x = v - 20 
    max_x = v + 20
    td = Multiplet(v , I, [(J, r)]) 
    grafica = mplplot(td.peaklist(), points=1000, w=0.5, limits=(min_x, max_x))
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

def det_width (xx, yy):

    # create a spline of the multiplet
    spline = UnivariateSpline(xx, yy-np.max(yy)/5, s=0)
    raices = spline.roots() # find the roots

    r1 = raices[0]
    r2 =raices[-1]

    pl.plot(xx, yy)
    pl.axvspan(r1, r2, facecolor='g', alpha=0.5)
    pl.xlabel("Desplazamiento químico (Hz)")
    pl.ylabel("Intensidad relativa")
    pl.show()

    print(f"Ancho de señal a media altura:\n valor minimo: {r1} Hz, valor maximo: {r2} Hz")
    
    w = r2-r1
    print(f"Ancho total: {w} Hz")

    return w, r1, r2

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


def central_signal (yy, xx, r2, r1):
    aa = np.array(yy)
    ss = np.array(xx) #shape: (1000,)
    #subsetter = np.where((ss >= int(r1)) & (ss <= int(r2)))
    subsetter = np.where((ss <= int(r2 +10)) & (ss >= int(r1 -10)))



    only_noise = np.delete(aa, subsetter)
    for i in range (len(only_noise)):
        if only_noise[i] < 0:
            only_noise[i] = 0
        
        else :
            only_noise [i]=i
    #print('solo señal: ',only_noise)
    #plt.plot(only_noise)
    #plt.show()
    return only_noise #return an array with only signal noise


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
def integrar(yy, intervalo=60, m=8):
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

#Confirmacion subarmonicos 
def Armonics (x, integ):
    armonic1 = x
    if armonic1 in integ: 
        print(f"armonico coincide perfectamente: {armonic1}")
        return armonic1 
    
    elif armonic1 + 1 in integ:
        print(f"armonico coincide +1: {armonic1 + 1}") 
        
        return armonic1 + 1

    elif armonic1 + 2 in integ: 
        print(f"armonico coincide +2: {armonic1 + 2}")
        
        return armonic1 + 2
    
    elif armonic1 - 1 in integ: 
        print(f"armonico coincide -1: {armonic1 - 1}")
        return armonic1 - 1
    
    elif armonic1 - 2 in integ:
        print(f"armonico coincide -2: {armonic1 -2}") 
        return armonic1 - 2
    
    else:
        print("subarmonico no confirmado")
        return 0
    

#señal = multiplet(1200, 1, 3.0, 1)
#archiv_txt(señal[0], señal[1])


yy, a, b = leer_archivo('SeñalesExperimentales/ha_08.slc')

iz = 0
de = len(yy) - 1

# Generar la secuencia xx
paso_hz = abs(a-b)/len(yy)
xx = [i*paso_hz+min([a, b]) for i in range(0, len(yy))]

#Determinando ancho de señal y relacion señal/ruido
w, r1, r2 = det_width(xx, yy)
noise = exp_noise(xx, yy, r1, r2)
mse, rmse, s_n = Noise(yy, noise)


#Redefiniendo el arreglo en y
yy = yy[iz:de]
xx = xx[iz:de]
nuevo_paso_hz = (xx[-1]-xx[0])/len(yy)

#redefiniendo otra ves yy con solo señal central
#y = central_signal(yy, xx, r2, r1)



#CORRECION LINEA BASE
#posiciones
##USING BITWISE OP
R2 = [p for p, v in enumerate(xx) if (v >= int(r2)) & (v <= int(r2+1))] 
R1 = [p for p, v in enumerate(xx) if (v >= int(r1)) & (v <= int(r1+1))] 

print(f"R2: {R2} y R1: {R1}")

pendiente = ((yy[len(yy)-1])- yy[0])/(len(yy)- 0)
ordenada_o = pendiente*xx[0] - yy[0]
print("pendiente: ", pendiente)
print("yy final: ", yy[len(yy)-1])



y = np.zeros(R2[0]-R1[0]-1)
for i in range(0, R2[0]-R1[0]-1):
    y[i] = yy[R1[0]+i] - (pendiente*i)

print("Minimo: ", min(y))
y = y - min(y)
plt.figure(figsize=(20,10))
#plt.plot(yy[R1[0]:R2[0]], color = "red")
#plt.plot(range(len(yy)), [pendientee(i) for i in range(len(yy))], color = "red")
plt.plot(yy, color = "blue")
plt.xlabel("Desplazamiento químico (Hz)")
plt.ylabel("Intensidad relativa")
plt.plot(y, color = "green")
plt.show()

#nueva escala
paso_hz_ = abs(R1[0]-R2[0])/len(y)
x = [i*paso_hz_+min([R1[0], R2[0]]) for i in range(0, len(y))]


# La escala en X de la siguiente figura está en enteros. Utilizar paso_Hz para convertir a Hz
#intervalo = int(len(yy)/4)
intervalo = int(len(y)/4)
m = 164
integrs = integrar(y, intervalo, m)

plt.figure(figsize=(20,10))
plt.plot(integrs, marker = 'o')
plt.xlabel("no. de punto")
plt.show()

busqueda = int(intervalo/8)
minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo

minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido

print(f"valores minimos (+ ruido) en: {minimos}")

armonic1 = int((minimosR[-1])/3)

subarmos= Armonics(armonic1, minimos)

Jota = minimosR[-1] * paso_hz

distance = Jota - (subarmos * paso_hz)


#Seleccionar el mínimo deseado para que se determine la J.¶
print(f"Jota Determinada: {Jota}         Resolución Digital: {paso_hz}")
print(f"la distancia es: {distance} Hz")


