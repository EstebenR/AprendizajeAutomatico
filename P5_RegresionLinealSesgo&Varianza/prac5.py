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
	return (np.sum((H-Y.T)**2))/(2*m) + (Lambda/(2*m))*np.sum(Theta[1:] **2)

def gradiente(Theta,X,Y):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	return np.dot(H-Y.T,X)/m

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

def calculaError(X,y, reg, Xval, Yval):
	m = np.shape(X)[0]
	mValidacion = np.shape(Xval)[0]
	errorValidacion = np.zeros([m])
	errorEntrenamiento = np.zeros([m])
	Xval = np.hstack([np.ones([mValidacion,1]),Xval])
	for i in range(1,m+1):
		theta = np.zeros(np.shape(X)[1])
		result = optimize.minimize(all, theta, args = (X[0:i], y[0:i,0], reg), jac = True, method = 'TNC')
		errorEntrenamiento[i-1] = costeRegularizado(result.x,X[0:i],y[0:i],0)
		errorValidacion[i-1] = costeRegularizado(result.x,Xval,Yval,0)
	
	return errorEntrenamiento,errorValidacion

def plotRegresionLineal(X, y, result,mod=""):
	plt.clf()
	plt.plot(X[:,1],y,"x")

	minX = np.min(X[:,1])
	maxX = np.max(X[:,1])

	xvalsResultsMin = result.x[0]+result.x[1]*minX
	xvalsResultsMax = result.x[0]+result.x[1]*maxX

	plt.plot([minX,maxX],[xvalsResultsMin,xvalsResultsMax])
	plt.savefig("RegresionLineal"+mod)
	plt.close()

def plotRegresionPolinomial(X,y,result, medias, desviaciones, mod=""):
	plt.clf()
	plt.plot(X[:,1],y,"x",  c = 'red')

	lineX = np.arange(np.min(X),np.max(X),0.05)
	aux_x = (nuevosDatos(lineX,8)-medias) / desviaciones
	lineY = np.hstack([np.ones([len(aux_x),1]),aux_x]).dot(result.x)
	plt.plot(lineX, lineY, '-', c = 'blue')	
	
	plt.savefig("RegresionLineal"+mod)
	plt.close()

def plotError(eEntrenamiento,eValidacion,mod=""):
	plt.clf()
	plt.plot(np.linspace(1,11,12,dtype=int),eEntrenamiento,label="Entrenamiento")
	plt.plot(np.linspace(1,11,12,dtype=int),eValidacion,label="Validacion")
	plt.legend()

	plt.savefig("ErrorValidacion"+mod)
	plt.close()

def plotErrorPolinomial(eEntrenamiento,eValidacion,mod=""):
	plt.clf()
	plt.plot(np.linspace(1,12,12,dtype=int),eEntrenamiento,label="Entrenamiento")
	plt.plot(np.linspace(1,12,12,dtype=int),eValidacion,label="Validacion")
	plt.legend()

	plt.savefig("ErrorValidacionPoli"+mod)
	plt.close()

def nuevosDatos(X, p):
	print(np.shape(X))
	res = np.empty([np.shape(X)[0], p])
	print(np.shape(res))
	for i in range(p):
		res[:, i] = (X**(i+1)).ravel()
	return res

def normaliza_matriz(mat):
    medias = np.mean(mat,axis=0)
    desviaciones = np.std(mat,axis=0)
    mat_norm = (mat - medias)/desviaciones
    return mat_norm, medias, desviaciones

def calculaErroresPolinomial(XextendidoNor,y,XvalPolinomial,yval):
	eEntPoli, eValPoli = calculaError(XextendidoNor,y,0,XvalPolinomial,yval)
	plotErrorPolinomial(eEntPoli,eValPoli,"0")

	eEntPoli, eValPoli = calculaError(XextendidoNor,y,1,XvalPolinomial,yval)
	plotErrorPolinomial(eEntPoli,eValPoli,"1")
	
	eEntPoli, eValPoli = calculaError(XextendidoNor,y,50,XvalPolinomial,yval)
	plotErrorPolinomial(eEntPoli,eValPoli,"50")

	eEntPoli, eValPoli = calculaError(XextendidoNor,y,100,XvalPolinomial,yval)
	plotErrorPolinomial(eEntPoli,eValPoli,"100")

def lambdasOptimos(XvalPolinomial, XextendidoNor, yval, y):
	#Lambda optimo
	lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
	
	errorValidacion = []
	errorEntrenamiento = []
	XvalPolinomial = np.hstack([np.ones([np.shape(XvalPolinomial)[0],1]),XvalPolinomial])
	for i in lambdas:
		theta = np.zeros(np.shape(XextendidoNor)[1])
		result = optimize.minimize(all, theta, args = (XextendidoNor, y, i), jac = True, method = 'TNC')
		errorEntrenamiento.append(costeRegularizado(result.x,XextendidoNor,y,0))
		errorValidacion.append(costeRegularizado(result.x,XvalPolinomial,yval,0))

	plt.clf()
	plt.plot(lambdas,errorEntrenamiento,label="Entrenamiento")
	plt.plot(lambdas,errorValidacion,label="Validacion")
	plt.legend()
	plt.savefig("PruebasLambdas")

def errorConPrueba(theta, XextendidoNor, XTest,  y, yTest, medias, desviaciones):
	result = optimize.minimize(all, theta, args = (XextendidoNor, y, 3), jac = True, method = 'TNC')

	xTestPolinomial = nuevosDatos(XTest,8)
	xTestPolinomial = (xTestPolinomial - medias) / desviaciones
	xTestPolinomial = np.hstack([np.ones([np.shape(xTestPolinomial)[0],1]),xTestPolinomial])

	error = costeRegularizado(result.x,xTestPolinomial,yTest,0)
	print(error)

def main():
	Xorig, y, Xval, yval, XTest, yTest = loadData()
	reg = 0
	m = np.shape(Xorig)[0]

	X = np.hstack([np.ones([m,1]),Xorig])
	
	theta = np.array([[1],[1]])
	result = optimize.minimize(all, theta, args = (X, y, reg), jac = True, method = 'TNC')
	print(f"result : {result.x}")
	plotRegresionLineal(X,y,result)
	eEnt, eVal = calculaError(X,y, reg,Xval,yval)
	plotError(eEnt,eVal)

	#Polinomial
	Xextendido = nuevosDatos(Xorig, 8)
	XextendidoNor, medias, desviaciones = normaliza_matriz(Xextendido)
	XextendidoNor = np.hstack([np.ones([np.shape(XextendidoNor)[0],1]),XextendidoNor])

	theta = np.zeros(np.shape(XextendidoNor[1]))
	result = optimize.minimize(all, x0 = theta, args = (XextendidoNor, y, reg), jac = True, method = 'TNC')
	print(result.x)
	plotRegresionPolinomial(X,y,result,medias,desviaciones,"1")
		

	XvalPolinomial = nuevosDatos(Xval,8)
	XvalPolinomial = (XvalPolinomial-medias) / desviaciones

	calculaErroresPolinomial(XextendidoNor,y,XvalPolinomial,yval)

	lambdasOptimos(XvalPolinomial, XextendidoNor, yval, y)
	
	errorConPrueba(theta, XextendidoNor, XTest, y, yTest, medias, desviaciones)

main()