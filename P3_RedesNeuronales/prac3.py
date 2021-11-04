from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_tnc

def loadData():
	data = loadmat('ex3data1.mat')
	y = data['y']
	X = data['X'] 
	return X,y

def showRandom(X,amount):
	sample = np.random.choice(X.shape[0],amount)
	plt.imshow(X[sample,:].reshape(-1,20).T)
	plt.axis('off')
	plt.show()

def sigmoide(x):
	return 1/( 1 + np.exp(-x)) 

def gradiente(Theta,X,Y):
	m = np.shape(X)[0]
	return  np.dot((np.transpose(X)),(sigmoide(np.dot(X,Theta))-Y))/m

def costeLineal(Theta,X,Y):
	m = np.shape(X)[0]
	H = np.dot(X,Theta)
	
	parte1 = np.dot(np.log(sigmoide(H)),Y)
	parte2 = np.dot(1-Y, (np.log(1-sigmoide(H))))
	return -(parte1 + parte2)/m   

def costeRegularizada(Theta, X, Y, Lambda):
	m = np.shape(X)[0]
	return costeLineal(Theta, X, Y) + (Lambda/(2*m))*np.sum(Theta**2)

def gradienteRegularizado(Theta,X,Y,Lambda):
	m = np.shape(X)[0]

	grad = gradiente(Theta,X,Y)
	grad_0 = grad[0]
	j = grad + (Lambda/m)*Theta
	j[0] = grad_0
	return j

def thetasOptimos(X,y,reg):
	theta = np.zeros([10,np.shape(X)[1]])
	for i in range(10):
		if(i == 0):
			result = fmin_tnc(func=costeRegularizada,x0=theta[i],fprime=gradienteRegularizado,args=(X,(y==10)*1,reg),messages=0)
		else:
			result = fmin_tnc(func=costeRegularizada,x0=theta[i],fprime=gradienteRegularizado,args=(X,(y==i)*1,reg),messages=0)
		theta[i]=result[0]
	return theta

def oneVsAll(X,y,num_etiquetas,reg):
	X = np.hstack([np.ones([num_etiquetas,1]),X])

	theta = thetasOptimos(X,y,reg)

	results = np.zeros([np.shape(X)[0],np.shape(theta)[0]])
	for i in range(np.shape(theta)[0]):
		results[:,i] = sigmoide(np.dot(X,theta[i]))

	maxIndices = np.argmax(results,1)
	evaluacion(maxIndices,y)

def evaluacion(maxIndices,y):
	acertados = np.sum(maxIndices==(y%10))
	print(f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(maxIndices)[0]}%")

def regresion():
	X, y = loadData()
	y = y[:,0]
	m = np.shape(X)[0]
	oneVsAll(X,y,m,0.1)

regresion()
	