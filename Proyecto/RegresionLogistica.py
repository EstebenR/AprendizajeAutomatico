import numpy as np
from pandas.io.parsers import read_csv
from scipy.optimize import fmin_tnc

categories = ["E", "ET", "T", "M"]


def cargaDatos(file):
	table = read_csv(file, header=0,).to_numpy()
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


def thetasOptimos(X, y, reg):
	theta = np.zeros([len(categories), np.shape(X)[1]])
	for i in range(len(categories)):
		result = fmin_tnc(func=costeRegularizada, x0=theta[i], fprime=gradienteRegularizado, args=(
			X, y[:, i], reg), messages=0)
		theta[i] = result[0]
	return theta


def evaluacion(maxIndices, Y, reg):
	acertados = np.sum(maxIndices == np.where(Y[:] == 1))
	print(
		f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(maxIndices)[0]:.3f}% con lambda = {reg}")


def oneVsAll():
	X, Y = cargaDatos('Video_games_esrb_rating.csv')
	testX, testY = cargaDatos('test_esrb.csv')
	m = np.shape(X)[0]
	X = np.hstack([np.ones([m, 1]), X])
	testX = np.hstack([np.ones([np.shape(testX)[0], 1]), testX])

	trainX = X[:int(m*0.8), :]
	validateX = X[int(m*0.8):, :]

	trainY = Y[:int(m*0.8), :]
	validateY = Y[int(m*0.8):, :]

	mejorAcertados = 0
	regOpt = 0
	thetasOpt = []

	for reg in np.arange(0.01, 3, 0.01):
		theta = thetasOptimos(trainX, trainY, reg)

		results = np.zeros([np.shape(validateX)[0], np.shape(theta)[0]])
		for i in range(np.shape(theta)[0]):
			results[:, i] = sigmoide(np.dot(validateX, theta[i]))

		maxIndices = np.argmax(results, 1)
		acertados = np.sum(maxIndices == np.where(validateY[:] == 1))
		if(acertados > mejorAcertados):
			regOpt = reg
			thetasOpt = theta
			mejorAcertados = acertados

	results = np.zeros([np.shape(testX)[0], np.shape(thetasOpt)[0]])
	for i in range(np.shape(thetasOpt)[0]):
		results[:, i] = sigmoide(np.dot(testX, thetasOpt[i]))

	maxIndices = np.argmax(results, 1)
	evaluacion(maxIndices, testY, regOpt)

	#TODO Grafica de lambd


oneVsAll()
