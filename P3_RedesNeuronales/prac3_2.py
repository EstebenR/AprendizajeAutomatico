from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc

def loadWeights():
	weights = loadmat('ex3weights.mat')
	theta1, theta2 = weights['Theta1'], weights['Theta2']
	# Theta1 es de dimensión 25 x 401
	# Theta2 es de dimensión 10 x 26
	return theta1, theta2

def loadData():
	data = loadmat('ex3data1.mat')
	y = data['y']
	X = data['X'] 
	return X,y

def sigmoide(x):
	return 1/( 1 + np.exp(-x)) 

def propagacion():
	X,y = loadData()
	m = np.shape(X)[0]
	X = np.hstack([np.ones([m,1]),X])
	theta1, theta2 = loadWeights()

	Z2 = np.dot(X, theta1.T)
	A2 = np.hstack([np.ones([m,1]),sigmoide(Z2)])
	Z3 = np.dot(A2, theta2.T)
	alpha3 = sigmoide(Z3)
	maxIndices = np.argmax(alpha3,axis=1)
	#Como maxIndices marca el indice, pero en matlab se indexa desde 1 hay que sumar uno al resultado de maxIndices para obtener el resultado correcto
	maxIndices = maxIndices+1
	acertados = np.sum(maxIndices==y.ravel())
	print(f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(alpha3)[0]}%")

propagacion()