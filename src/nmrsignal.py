"""Represent signals output by an NMR system"""

import random

import numpy as np

def ppm_to_hz(ppm:float, frequency_mhz:float) -> float:
    "Convert values from ppm to Hz produced by nmr"
    return ppm*frequency_mhz / (10**6)

class Signal:
    """Represents an NMR signal"""

    def  __init__(self,  y_values, min_hertz_value, max_hertz_value):
        self.y_values = y_values
        self.min_hertz = min_hertz_value
        self.max_hertz = max_hertz_value
        # Assumes that the y_values were taken uniformly over the hertz range
        self.point_count = len(y_values)
        self.gap = (self.max_hertz - min_hertz_value) / self.point_count
        self.x_values = [self.min_hertz + x*self.gap for x in range(0,self.point_count)]
       
    def __str__(self):
        return f"min: {self.min_hertz}    max: {self.max_hertz}\nx:{self.x_values}\ny:{self.y_values}"

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
            
            return Signal(yy, float(b), float(a))
    
    @staticmethod
    def read_file_mnova_csv(filename, frequency_mhz):
        """Reads a sgnal generated using the mnvoa export as nmr csvv option"""
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            xy_pairs = []
            for l in lines:
                (xs, ys) = l.split()
                (x,y) = (float(xs), float(ys))
                pair = (ppm_to_hz(x, frequency_mhz),y)
                xy_pairs.append(pair)
        
        # A rather untasteful unzip
        (x_values, y_values) = zip(*xy_pairs)
        x_values = list(x_values)
        y_values = list(y_values)

        sig = Signal(y_values, x_values[0]*10**6, x_values[-1]*10**6)
        sig.x_values = x_values
        return sig            

    def write_file_mnova_csv(self, filename, frequency_mhz):
        "Write a file in the mnova csv export. Intended for our own use"
        output_str = ""
        #not super sure I am managing the frequencies correctly
        #Also note that the measurements are assumed to be uniform over X
        for i, val in enumerate(self.y_values):
            output_str += f"{i*self.gap/frequency_mhz}\t{val*10**6 / frequency_mhz}\t\n"
        with open(filename, "w") as f:
            f.write(output_str)


def gen_random_signal(point_count, min_start_height, max_start_height, min_jump, max_jump):
    """Generates a random signal that looks like an nmr produced signal"""
    x_coords = list(range(0,point_count))
    y_coords = []

    for x in range(0, len(x_coords)):
        if x==0:
            y_coords.append(random.uniform(min_start_height, max_start_height))
        else:
            next_y = random.uniform(min_jump, max_jump)
            next_y += y_coords[x-1]
            y_coords.append(next_y)

    return Signal(y_coords, 0, len(x_coords))
