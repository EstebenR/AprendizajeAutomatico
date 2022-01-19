import numpy as np
from pandas.io.parsers import read_csv
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

categories = ["E", "ET", "T", "M",]


def cargaDatos(file):
	table = read_csv(file, header=0,).to_numpy()
	X = table[:, 1:-1]
	# Hay 7 tipos de ratings que vienen codificados con un string, asi que los codificamos como ints
	Y = np.empty([np.shape(table)[0], 4])
	for index, cat in enumerate(categories):
		col = table[:, -1] == cat
		Y[:, index] = col

	return X.astype(int), Y.astype(int)

def entrenamiento(X, y, xVal, yVal, trainValues):
	numTrainValues = len(trainValues)
	scoreOpt = 0
	svmOpt = 0
	cOpt = 0
	sOpt = 0
	scores = np.empty([numTrainValues,numTrainValues])
	for i in range(numTrainValues):
		cVal = trainValues[i]
		for j in range(numTrainValues):
			sigma = trainValues[j]
			svm = SVC(kernel='rbf', C=cVal, gamma=1/(2 * sigma**2))
			svm.fit(X, y.ravel())
			accuracy = accuracy_score(yVal, svm.predict(xVal))
			scores[i,j] = accuracy
			if(accuracy > scoreOpt):
				svmOpt = svm
				scoreOpt = accuracy
				cOpt = cVal
				sOpt = sigma


	return svmOpt,cOpt,sOpt,scoreOpt,scores


def muestraGrafica(X, Y, val):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Z = np.meshgrid(X,X)
	ax.plot_surface(X, Z, Y, cmap=cm.jet)
	ax.set_xlabel("sigma")
	ax.set_ylabel("C")
	ax.set_zlabel("acertados")
	plt.savefig(f"acertadosSVM{val}.png")
	plt.clf()


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
	sOpt = []
	scoreOpt = []
	scoresTotal = []
	
	for i in range(4):
		_svmOpt, _cOpt, _sOpt, _scoreOpt, scores = entrenamiento(trainX, trainY[:,i], validateX, validateY[:,i], trainValues)
		svmOpt.append(_svmOpt)
		cOpt.append(_cOpt)
		sOpt.append(_sOpt)
		scoreOpt.append(_scoreOpt)
		scoresTotal.append(scores)

	testScores = 0
	for i in range(4):
		testScores += svmOpt[i].score(testX,testY[:,i])
	
	
	print(f"Optimum Cs: {cOpt}")
	print(f"Optmimum sigmas: {sOpt}")
	print(f"Average precision: {testScores/4*100}%")
	for i, score in enumerate(scoresTotal):
		muestraGrafica(trainValues,score,i)

SVM()