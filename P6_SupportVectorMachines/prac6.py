from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt
from process_email import email2TokenList
from get_vocab_dict import getVocabDict

def loadData(num):
	data = loadmat('ex6data'+str(num)+'.mat')
	X = data['X']
	y = data['y']
	if num != 3:
		return X, y
	elif num == 3:
		XVal = data['Xval']
		yVal = data['yval']
		return X,y,XVal, yVal

def visualize_boundary(X, y, svm, file_name):
	x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
	x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
	x1, x2 = np.meshgrid(x1, x2)
	yp = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)
	pos = (y == 1).ravel()
	neg = (y == 0).ravel()
	plt.figure()
	plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
	plt.scatter(
	X[neg, 0], X[neg, 1], color='yellow', edgecolors='black', marker='o')
	plt.contour(x1, x2, yp)
	plt.savefig(file_name)
	plt.close()

def entrenamiento(X,y,xVal,yVal,trainValues):
	numTrainValues = len(trainValues)
	scoreOpt = 0
	svmOpt = 0
	for i in range(numTrainValues):
		cVal = trainValues[i]
		for j in range(numTrainValues):
			sigma = trainValues[j]
			svm = SVC(kernel='rbf',C=cVal,gamma=1/(2 * sigma**2))
			svm.fit(X,y.ravel())
			accuracy = accuracy_score(yVal, svm.predict(xVal))
			if(accuracy > scoreOpt):
				svmOpt = svm
				scoreOpt = accuracy

	return svmOpt

def parte1():
	X,y = loadData(1)
	cVal = 1.0
	svm = SVC(kernel='linear', C=cVal)
	svm.fit(X, y.ravel())

	visualize_boundary(X,y,svm,"Boundary_C"+str(cVal)+".png")

	X,y = loadData(2)
	sigma = 0.1
	svm = SVC(kernel='rbf',C=cVal,gamma=1/(2 * sigma**2))
	svm.fit(X,y.ravel())
	
	visualize_boundary(X,y,svm,"BoundaryGauss_C"+str(cVal)+".png")

	X,y,xVal,yVal = loadData(3)
	trainValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	svmOpt = entrenamiento(X,y,xVal,yVal,trainValues)
	visualize_boundary(X,y,svmOpt,"ParametrosOptimos.png")


def parte2():
	email_contents = open('spam/0001.txt' ,  'r', encoding='utf-8', errors='ignore').read()
	email = email2TokenList(email_contents)
	print(email)
	vocab = getVocabDict()

parte2()