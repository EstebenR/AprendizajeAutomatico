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
	X = np.hstack([np.ones([np.shape(X)[0],1]),X])
	theta1, theta2 = loadWeights()

	zeta = np.apply_along_axis(np.dot,0,theta1,X)
	alpha2 = sigmoide(zeta)
	zeta3 = np.apply_along_axis(np.dot,0,theta2,alpha2)
	alpha3 = sigmoide(zeta3)
	maxIndices = np.argmax(alpha3)
	acertados = np.sum(maxIndices==y)
	print(f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(alpha3)[0]}%")

propagacion()