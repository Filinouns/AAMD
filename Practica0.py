import time
import math as mt
import numpy as np
import matplotlib.pyplot as plt
#from numpy import *
#from math import *

#cuadratica
def f1(x):
    return 2*(x**2) + 5*x - 2

#lineal
def f2(x):
    return 4*x + 1

def f3(x):
    return mt.sin(x)

def f4(x):
    return mt.cos(x)

def dot_product(x1, x2):
    """Calcula el producto escalar con un bucle 
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    dot = 0
    for i in range (len(x1)):
        dot += x1[i] * x2[i]
    toc = time.process_time()
    return 1000 * (toc-tic)

def fast_dot_product(x1, x2):
    """Calcula el producto escalar vectorizado
    y devuelve el tiempo en milisegundos"""
    tic = time.process_time()
    #dot = np.dot(x1, x2)
    toc = time.process_time()
    return 1000 * (toc - tic)

def compara_tiempos():
    sizes = np.linspace(100, 10000000, 20)
    times_dot = []
    times_fast = []
    for size in sizes:
        x1 = np.random.uniform(1, 100, int(size))
        x2 = np.random.uniform(1, 100, int(size))
        times_dot += [dot_product(x1, x2)]
        times_fast += [fast_dot_product(x1, x2)]

    plt.figure()
    plt.scatter(sizes, times_dot, c='red', label='bucle')
    plt.scatter(sizes, times_fast, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')

#---Program---
x = -5
y = 15
r = np.arange(x, y, 0.1)

#plt.plot(r, [f1(i) for i in r])
#plt.plot(r, [f2(i) for i in r])
plt.plot(r, [f3(i) for i in r])
plt.plot(r, [f4(i) for i in r])

plt.axhline(0, color = "black")
plt.axvline(0, color = "black")

plt.xlim(0, 5)
plt.ylim(-1, 1)

plt.savefig("Practica0.png")

plt.show()
#compara_tiempos()