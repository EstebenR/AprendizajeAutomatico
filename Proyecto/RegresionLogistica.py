from re import X
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures

categories = ["RP", "EC", "E", "ET", "T", "M", "A"]


def cargaDatos():
	table = read_csv('Video_games_esrb_rating.csv', header=0,).to_numpy()
	X = table[:, 1:-1]
	# Hay 7 tipos de ratings que vienen codificados con un string, asi que los codificamos como ints
	Y = np.empty([np.shape(table)[0], 7])
	for index, cat in enumerate(categories):
		col = table[:, -1] == cat
		Y[:, index] = col

	return X.astype(int), Y.astype(int)


def sigmoide(zeta):
	return 1 / (1+np.exp(-zeta))


def costeLineal(Theta, X, Y):
	m = np.shape(X)[0]
	H = np.dot(X, Theta)

	parte1 = np.dot(np.log(sigmoide(H)), Y)
	parte2 = np.dot(1-Y, (np.log(1-sigmoide(H))))
	return -(parte1 + parte2)/m


def gradiente(Theta, X, Y):
	m = np.shape(X)[0]
	return np.dot((np.transpose(X)), (sigmoide(np.dot(X, Theta))-Y))/m


def costeRegularizada(Theta, X, Y, Lambda):
	m = np.shape(X)[0]
	return costeLineal(Theta, X, Y) + (Lambda/(2*m))*np.sum(Theta**2)


def gradienteRegularizado(Theta, X, Y, Lambda):
	m = np.shape(X)[0]

	grad = gradiente(Theta, X, Y)
	grad_0 = grad[0]
	j = grad + (Lambda/m)*Theta
	j[0] = grad_0
	return j

def thetasOptimos(X,y,reg):
	theta = np.zeros([7,np.shape(X)[1]])
	for i in range(7):
		result = fmin_tnc(func=costeRegularizada,x0=theta[i],fprime=gradienteRegularizado,args=(X,y[:,i],reg),messages=0)
		theta[i]=result[0]
	return theta

def evaluacion(maxIndices,Y):
	acertados = np.sum(maxIndices==np.where(Y[:]==1))
	print(f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(maxIndices)[0]}%")

def oneVsAll():
	print(categories)
	X,Y = cargaDatos()
	X = np.hstack([np.ones([np.shape(X)[0],1]),X])

	reg = 2
	theta = thetasOptimos(X,Y,reg)

	results = np.zeros([np.shape(X)[0],np.shape(theta)[0]])
	for i in range(np.shape(theta)[0]):
		results[:,i] = sigmoide(np.dot(X,theta[i]))

	maxIndices = np.argmax(results,1)
	print(np.shape(maxIndices))
	print(maxIndices)
	evaluacion(maxIndices,Y)


oneVsAll()