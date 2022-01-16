import numpy as np
from pandas.io.parsers import read_csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

def entrenamiento(X, y, xVal, yVal, trainValues):
	numTrainValues = len(trainValues)
	scoreOpt = 0
	svmOpt = 0
	cOpt = 0
	for i in range(numTrainValues):
		cVal = trainValues[i]
		for j in range(numTrainValues):
			sigma = trainValues[j]
			svm = SVC(kernel='rbf', C=cVal, gamma=1/(2 * sigma**2))
			svm.fit(X, y)
			accuracy = accuracy_score(yVal, svm.predict(xVal))
			if(accuracy > scoreOpt):
				svmOpt = svm
				scoreOpt = accuracy
				cOpt = cVal


	return svmOpt,cOpt,scoreOpt

def SVM():
	X, Y = cargaDatos()
	m = np.shape(X)[0]

	trainX = X[:int(m*0.6), :]
	testX = X[int(m*0.6):int(m*0.8), :]
	validateX = X[int(m*0.8):, :]

	trainY = Y[:int(m*0.6), :]
	testY = Y[int(m*0.6):int(m*0.8), :]
	validateY = Y[int(m*0.8):, :]

	print("starting training")

	trainValues = np.arange(0,30,1)

	svmOpt = []
	cOpt = []
	scoreOpt = []
	
	for i in range(7):
		_svmOpt, _cOpt, _scoreOpt = entrenamiento(trainX, trainY[:,i], validateX, validateY[:,i], trainValues)
		svmOpt[i] = _svmOpt
		cOpt[i] = _cOpt
		scoreOpt[i] = _scoreOpt
	

	testScores = 0
	for i in range(7):
		testScores += svmOpt.score(testX,testY[:,i])
	
	
	print(f"Optimum Cs: {cOpt}")
	#print(f"Error: {1-scoreOpt}")
	print(f"Average recision: {testScores/7*100}%")

SVM()