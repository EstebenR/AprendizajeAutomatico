from numpy.core.fromnumeric import shape
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from scipy.optimize import fmin_tnc


def loadData():
	data = loadmat('ex5data1.mat')
	X = data['X'] 
	y = data['y']
	Xval = data["Xval"]
	yval = data["yval"]
	Xtest = data["Xtest"]
	Ytest = data["ytest"]
	return X,y,Xval,yval,Xtest,Ytest

def costeRegularizado(Theta, X, Y, Lambda):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	return (np.sum((H-Y)**2))/(2*m) + (Lambda/(2*m))*np.sum(Theta[1:] **2)

def gradiente(Theta,X,Y):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	return np.matmul(np.transpose(X),H-Y)/m

def gradienteRegularizado(Theta,X,Y,Lambda):
	m = np.shape(X)[0]

	grad = gradiente(Theta,X,Y)
	grad_0 = grad[0]
	j = grad + (Lambda/(m))*Theta
	j[0] = grad_0
	return j

def all(theta, X, y, reg):
	return costeRegularizado(theta,X,y,reg), gradienteRegularizado(theta, X, y, reg)

def hipothesis(X,theta):
	return np.sum(theta*X,1)

def calculaError(X,y,theta, reg, Xval, Yval):
	m = np.shape(X)[0]
	mValidacion = np.shape(Xval)[0]
	errorValidacion = np.zeros([m])
	errorEntrenamiento = np.zeros([m])
	Xval = np.hstack([np.ones([mValidacion,1]),Xval])
	print("---")
	for i in range(1,m):
		result = optimize.minimize(all, theta, args = (X[0:i], y[0:i,0], reg), jac = True, method = 'TNC')
		errorEntrenamiento[i] = (np.sum((hipothesis(X[0:i],result.x)-y[0:i,0])**2))/(2*i)
		errorValidacion[i] = (np.sum((hipothesis(Xval,result.x)-Yval[:,0])**2))/(2*mValidacion)
	
	return errorEntrenamiento,errorValidacion

def plotRegresionLineal(X, y, result):
	plt.clf()
	plt.plot(X[:,1],y,"x")

	minX = np.min(X[:,1])
	maxX = np.max(X[:,1])

	xvalsResultsMin = result.x[0]+result.x[1]*minX
	xvalsResultsMax = result.x[0]+result.x[1]*maxX

	plt.plot([minX,maxX],[xvalsResultsMin,xvalsResultsMax])
	plt.show()

def plotError(eEntrenamiento,eValidacion):
	plt.clf()
	eje = np.arange(1,np.shape(eEntrenamiento)[0])
	plt.plot(eje,eEntrenamiento[1:])
	plt.plot(eje,eValidacion[1:])

	plt.show()

def nuevosDatos(X, p):
	res = np.zeros([np.shape(X)[0], p])
	res[:, 0] = X
	for i in range(1, p):
		res[:, i] = X**(i+1)
	return res

def normaliza_matriz(mat):
    medias = mat.mean(axis=0)
    desviaciones = mat.std(axis=0) + 1e-7
    mat_norm = (mat[:]-medias)/desviaciones
    return mat_norm, medias, desviaciones

def main():
	X,y, Xval, yval, XTest, yTest = loadData()
	reg = 0
	m = np.shape(X)[0]
	X = np.hstack([np.ones([m,1]),X])


	Xextendido = nuevosDatos(X[:,1], 5)
	Xextendido = np.hstack([np.ones([m,1]),Xextendido])
	XextendidoNor, medias, desviaciones = normaliza_matriz(Xextendido)

	print(np.shape(XextendidoNor),np.shape(medias), np.shape(desviaciones))

	theta = np.array(np.shape(XextendidoNor))
	result = optimize.minimize(all, theta, args = (XextendidoNor, y[:,0], reg), jac = True, method = 'TNC')
	print(result.x)
	#theta = np.array([[1],[1]])
	#print(costeRegularizado(theta,X,y,1))
	#print(gradienteRegularizado(theta,X,y,1))
	#result = optimize.minimize(all, theta, args = (X, y[:,0], reg), jac = True, method = 'TNC')
	#plotRegresionLineal(X,y,result)
	#eEnt, eVal = calculaError(X,y,theta, reg,Xval,yval)
	#plotError(eEnt,eVal)

main()