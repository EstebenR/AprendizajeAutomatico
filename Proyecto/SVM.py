import numpy as np
from pandas.io.parsers import read_csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

categories = ["E", "ET", "T", "M",]


def cargaDatos(file):
	table = read_csv(file, header=0,).to_numpy()
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
			svm.fit(X, y.ravel())
			accuracy = accuracy_score(yVal, svm.predict(xVal))
			if(accuracy > scoreOpt):
				svmOpt = svm
				scoreOpt = accuracy
				cOpt = cVal


	return svmOpt,cOpt,scoreOpt

def SVM():
	X, Y = cargaDatos('Video_games_esrb_rating.csv')
	testX, testY = cargaDatos('test_esrb.csv')
	m = np.shape(X)[0]
	

	trainX = X[:int(m*0.8), :]
	validateX = X[int(m*0.8):, :]

	trainY = Y[:int(m*0.8), :]
	validateY = Y[int(m*0.8):, :]

	print("starting training")

	trainValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

	svmOpt = []
	cOpt = []
	scoreOpt = []
	
	for i in range(4):
		_svmOpt, _cOpt, _scoreOpt = entrenamiento(trainX, trainY[:,i], validateX, validateY[:,i], trainValues)
		svmOpt.append(_svmOpt)
		cOpt.append(_cOpt)
		scoreOpt.append(_scoreOpt)
	

	testScores = 0
	for i in range(4):
		testScores += svmOpt[i].score(testX,testY[:,i])
	
	
	print(f"Optimum Cs: {cOpt}")
	#print(f"Error: {1-scoreOpt}")
	print(f"Average recision: {testScores/7*100}%")

SVM()