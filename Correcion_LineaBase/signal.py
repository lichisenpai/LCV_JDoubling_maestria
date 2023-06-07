"""Represent signals output by an NMR system"""

import numpy as np

class Signal:
    """Represents an NMR signal"""

    def  __init__(self,  y_values, min_hertz_value, max_hertz_value):
        self.y_values = y_values
        self.min_hertz = min_hertz_value
        self.max_hertz = max_hertz_value
        # Assumes that the y_values were taken uniformly over the hertz range
        self.point_count = len(y_values)
        point_gap = (self.max_hertz - min_hertz_value) / self.point_count
        self.x_values = [self.min_hertz + x*point_gap for x in range(0,self.point_count)]
       
    def __str__(self):
        return f"min: {self.min_hertz}    max: {self.max_hertz}\n{self.y_values}"

    @staticmethod
    def leer_archivo(nombre):
        """Read an .slc file as a Signal"""
        #Hay que analizar las primeras 3 renglones por separado.
        #a, b seran los extremos del intervalo en Hz
        
        with open(nombre, "r", encoding="utf-8") as arch:
            contenido = arch.read()
            contenido = contenido.split("\n")
            tam = int(contenido[0].split(": ")[1])
            [a, b] = contenido[2].split(": ")[1].split(" ")

            yy = np.zeros(tam)
            for i in range(0, tam):
                yy[i] = (float(contenido[i+3]))
            
            return Signal(yy, float(a), float(b))
