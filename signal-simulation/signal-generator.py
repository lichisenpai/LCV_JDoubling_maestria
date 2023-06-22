
class NAME:
    def __init__(self) -> None:
        self.calc = ....
        self.Error  =
    
    def to_csv(self):
        ....

# calc := J outputed by jdoubling
# Error:= diff between theorical expected J and calc
# Error2 := quadratic error  <---- possibly useless
# SubHarmonics := second minimum on integral graph when read right to left
# DistHz := Distance between the first and second minimum in the integral graph
## reminder, the first minimum is the J itself
# S_n := signal to noise ratio
# signalWidth := Width of the signal in hertz








##############################################################################

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
    intensidades = multiplete[1] 
    desplazamiento = multiplete[0] 
    señalCruido = ruido + intensidades 
    mse, rmse, s_n = Noise(intensidades, ruido) 
    #Grafica 
    """plt.plot(desplazamiento, señalCruido)
    plt.title(f"Relación Señal/Ruido: {s_n}")
    plt.xlabel("Desplazamiento (Hz)")
    plt.ylabel("Intensidades")
    plt.show()"""
    
    archiv_txt(desplazamiento, señalCruido)

    #JDoubling
    yy, a, b = leer_archivo('Prueba.slc')

    iz = 0
    de = len(yy) - 1

    # Generar la secuencia xx
    paso_hz = abs(a-b)/len(yy)
    xx = [i*paso_hz+min([a, b]) for i in range(0, len(yy))]

    #Redefiniendo el arreglo en y
    yy = yy[iz:de]
    xx = xx[iz:de]
    nuevo_paso_hz = (xx[-1]-xx[0])/len(yy)

    # La escala en X de la siguiente figura está en enteros. Utilizar paso_Hz para convertir a Hz
    intervalo = int((J / paso_hz) * 1.3)
    m = 164
    integrs = integrar(yy, intervalo, m)

    """plt.figure(figsize=(20,10))
    plt.plot(integrs, marker = 'o')
    plt.show()"""

    busqueda = int(intervalo/7)
    minimosR = argrelextrema(integrs, np.less, order=busqueda, mode= 'wrap')[0]#busca el minimo mas minimo

    minimos = argrelextrema(integrs, np.less, mode='wrap')[0] #me da todos los minimos con ruido

    print(f"valores minimos (+ ruido) en: {minimos}")

    
    armonic1 = int((minimosR[-1])/3)

    subarmos= Armonics(armonic1, minimos)
    subarmosHz = subarmos * paso_hz

    Jota = minimosR[-1] * paso_hz

    calc.append(Jota) 
    SubHarmonics.append(subarmosHz)
    S_n.append(s_n)

    #Seleccionar el mínimo deseado para que se determine la J.¶
    print(f"Jota: {Jota}         Resolución Digital: {paso_hz}")


Error = list(np.array(jotas)-np.array(calc)) 
Error2 = list(map(lambda x: np.square(x), Error))
DistHz = list(np.array(calc)-np.array(SubHarmonics)) #distancia entre J y el primer subarmónico 


for a, b, c, d, e, f in zip(calc, Error, SubHarmonics, DistHz, S_n, Error2):
    new_entry = new_data(a, b, c, d, e, f)
    Jota_12_0Hz.append(new_entry)
escritura_json("J12_0yW0_5ruido")                