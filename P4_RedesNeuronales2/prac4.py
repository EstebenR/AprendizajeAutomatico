from numpy.core.fromnumeric import shape
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc
from scipy.optimize.optimize import OptimizeResult
from displayData import *
from checkNNGradients import checkNNGradients

def loadWeights():
	weights = loadmat('ex4weights.mat')
	theta1, theta2 = weights['Theta1'], weights['Theta2']
	return theta1, theta2

def loadData():
	data = loadmat('ex4data1.mat')
	y = data['y']
	X = data['X'] 
	return X,y

def showRandom(X,amount):
	sample = np.random.choice(X.shape[0],amount)
	res = displayData(X[sample])

def sigmoide(x):
	return 1/( 1 + np.exp(-x)) 

def propagacion(X,theta1,theta2):
	m = np.shape(X)[0]
	A1 = np.hstack([np.ones([m,1]),X])
	Z2 = np.dot(A1, theta1.T)
	A2 = np.hstack([np.ones([m,1]),sigmoide(Z2)])
	Z3 = np.dot(A2, theta2.T)
	A3 = sigmoide(Z3)
	return A1, A2, A3

def costeSinRegularizar(X,y, t1,t2):
	m = np.shape(X)[0]
	a1,a2,h = propagacion(X,t1,t2)
	part1 = y*np.log(h)
	part2 = (1-y)*np.log(1-h)
	return (-1/m)*np.sum(part1+part2)

def costeRegularizado(X,y, t1,t2, reg):
	m = np.shape(X)[0]
	
	aux = costeSinRegularizar(X, y, t1, t2)
	otra = (reg/(2*m)) * (np.sum(np.power(t1,2)) +np.sum(np.power(t2, 2)))
	
	return aux + otra

def resizeY(y,numLabels):
	m = len(y)
	
	y = (y-1)
	y_onehot = np.zeros((m,numLabels))

	for i in range(m):
		y_onehot[i][y[i]] = 1

	return y_onehot

def terminoRegularizacion(gradiente, m, reg, theta):
	columnaGuardada = gradiente[:,0]
	gradiente = gradiente + (reg/m)*theta
	gradiente[:,0] = columnaGuardada
	return gradiente

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	m = np.shape(X)[0]

	Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)] , (num_ocultas, (num_entradas+1)))
	Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):] , (num_etiquetas, (num_ocultas+1)))

	Delta1 = np.zeros_like(Theta1)
	Delta2 = np.zeros_like(Theta2)

	A1, A2, H = propagacion(X,Theta1, Theta2)

	for t in range(m):
		a1t = A1[t,:]
		a2t = A2[t,:]
		ht = H[t,:]
		yt = y[t]

		d3t = ht -yt
		d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t))

		Delta1 = Delta1 + np.dot(d2t[1:,np.newaxis], a1t[np.newaxis,:])
		Delta2 = Delta2 + np.dot(d3t[:,np.newaxis],a2t[np.newaxis,:])

	gradiente1 = Delta1/m
	gradiente2 = Delta2/m

	gradiente1 = terminoRegularizacion(gradiente1, m, reg,Theta1)
	gradiente2 = terminoRegularizacion(gradiente2, m, reg,Theta2)

	return (costeRegularizado(X,y,Theta1,Theta2,reg),np.concatenate([np.ravel(gradiente1),np.ravel(gradiente2)]))



def main():
	X, y = loadData()
	T1, T2 = loadWeights() #Para comparar

	y = resizeY(y.ravel(),10)
	params = np.concatenate([np.ravel(T1),np.ravel(T2)])
	
	cost, reg_param =backprop(params,np.shape(X)[1],25,10,X,y,1) 
	checkNNGradients(cost,reg_param)

main()