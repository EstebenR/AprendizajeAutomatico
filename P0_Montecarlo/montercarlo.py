import time
import numpy as np
from matplotlib import pyplot as plt

def maximoFuncion(fun, a, b):
    return np.amax(fun(np.linspace(a,b,1000)))

def integra_mc(fun, a, b, num_puntos=10000):
    porDebajo = 0
    maxFuncion = maximoFuncion(fun,a,b)
    tic = time.process_time()
    for i in range(num_puntos):
        ranX = (b-a) * np.random.random() + a
        ranY = maxFuncion * np.random.random()
        if (ranY < fun(ranX)):
            porDebajo += 1
    #return porDebajo/num_puntos*(b-a)*maxFuncion
    toc = time.process_time()
    return 1000*(toc-tic)

def fast_integra_mc(fun,a,b,num_puntos=10000):
    maxFuncion = maximoFuncion(fun,a,b) #Calculo del maximo de la funcion queda por fuera de nuestro calculo de tiempo
    tic = time.process_time()
    valoresX = (b-a) * np.random.rand(num_puntos) + a
    valoresY = maxFuncion * np.random.rand(num_puntos)
    valoresFuncion = func(valoresX)
    porDebajo = np.sum(valoresY<valoresFuncion)
    #return porDebajo/num_puntos*(b-a)*maxFuncion
    toc = time.process_time()
    return 1000 * (toc-tic)

def func(x):
    return x**2

def compara_tiempos():
    tams = np.linspace(100,1000000,50,dtype='int')
    tiemposNormal = []
    tiemposFast = []
    for tam in tams:
        tiemposNormal += [integra_mc(func,0,2,tam)]
        tiemposFast += [fast_integra_mc(func,0,2,tam)]

    plt.figure()
    plt.scatter(tams, tiemposNormal, c='red', label='bucle')
    plt.scatter(tams, tiemposFast, c='blue', label='vector')
    plt.legend()
    plt.savefig('grafica.png')

compara_tiempos()