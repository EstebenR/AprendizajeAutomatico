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
	j = grad + (Lambda/m)*Theta
	j[0] = grad_0
	return j

def all(theta, X, y, reg):
	return costeRegularizado(theta,X,y,reg), gradienteRegularizado(theta, X, y, reg)

def plot(X, y, result):
	plt.clf()
	plt.plot(X,y,"x")

	plt.show()

def main():
	X,y, Xval, yval, XTest, yTest = loadData()
	reg = 0
	m = np.shape(X)[0]
	X = np.hstack([np.ones([m,1]),X])
	theta = np.array([[1],[1]])
	print(costeRegularizado(theta,X,y,1))
	print(gradienteRegularizado(theta,X,y,1))
	result = optimize.minimize(all, theta, args = (X, y[:,0], reg), jac = True, method = 'TNC')
	print(result.x)
	plot(X,y,result)
	

main()