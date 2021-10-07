import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

def carga_csv(file_name):
	valores = read_csv(file_name,header=None).to_numpy()

	return valores.astype(float)

def ver_datos(X,Y):
	pos = np.where(Y == 1)
	pos2 = np.where(Y == 0)

	plt.scatter(X[pos,0],X[pos,1],marker='+',c='k')
	plt.scatter(X[pos2,0],X[pos2,1],c='orange')
	plt.savefig("regresionLogistica.png")

def sigmoide(zeta):
	return 1 / (1+np.exp(-zeta))

def coste(Theta,X,Y):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	
	parte1 = np.dot(np.log(sigmoide(H)),Y)
	parte2 = np.dot(1-Y, (np.log(1-sigmoide(H))))
	return -(parte1 + parte2)/m

def gradiente(Theta,X,Y):
	m = np.shape(X)[0]
	return  np.dot((np.transpose(X)),(sigmoide(np.dot(X,Theta))-Y))/m

def pinta_frontera_recta(X,Y,Theta):
	pos = np.where(Y == 1)
	pos2 = np.where(Y == 0)

	x1_min,x1_max = X[:,1].min(), X[:,1].max()
	x2_min,x2_max = X[:,2].min(), X[:,2].max()

	punto1 = Theta[0]+Theta[1]*x1_min+Theta[2]*x1_max
	punto2 = Theta[0]+Theta[1]*x1_max+Theta[2]*x2_min

	print(punto1)
	print(punto2)
	
	plt.plot([x1_min, punto1], [x1_max,punto2])

	plt.scatter(X[pos,1],X[pos,2],marker='+',c='k')
	plt.scatter(X[pos2,1],X[pos2,2],c='orange')
	plt.show()

def regresion_logistica(datos):
	X = datos[:,:-1]
	Y = datos[:,-1]
	m = np.shape(X)[0]
	X = np.hstack([np.ones([m,1]),X])

	Theta = np.zeros(np.shape(X)[1])

	result = fmin_tnc(func=coste,x0=Theta,fprime=gradiente,args=(X,Y),messages=0)
	theta_opt = result[0]

	print(theta_opt)
	print(coste(theta_opt,X,Y))

	pinta_frontera_recta(X,Y,theta_opt)


#ver_datos(carga_csv("ex2data1.csv"))
regresion_logistica(carga_csv("ex2data1.csv"))