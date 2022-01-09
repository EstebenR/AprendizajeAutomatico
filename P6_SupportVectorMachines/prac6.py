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
		return X, y, XVal, yVal


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


def parte1():
	X, y = loadData(1)
	cVal = 1.0
	svm = SVC(kernel='linear', C=cVal)
	svm.fit(X, y.ravel())

	visualize_boundary(X, y, svm, "Boundary_C"+str(cVal)+".png")

	X, y = loadData(2)
	sigma = 0.1
	svm = SVC(kernel='rbf', C=cVal, gamma=1/(2 * sigma**2))
	svm.fit(X, y.ravel())

	visualize_boundary(X, y, svm, "BoundaryGauss_C"+str(cVal)+".png")

	X, y, xVal, yVal = loadData(3)
	trainValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	svmOpt, cOpt,scoreOpt = entrenamiento(X, y, xVal, yVal, trainValues)
	visualize_boundary(X, y, svmOpt, "ParametrosOptimos.png")


def readEmails(folder, num, vocab):
	email_contents = open(folder + "/" + str(num).zfill(4) +
	                      '.txt',  'r', encoding='utf-8', errors='ignore').read()
	email = email2TokenList(email_contents)

	mailVector = np.zeros([len(vocab)])
	for word in email:
		if word in vocab:
			mailVector[vocab[word]-1] = 1

	return mailVector


def parte2():
	vocab = getVocabDict()

	numSpam = 500
	numEasyHam = 2551
	numHardHam = 250

	spamX = np.zeros([numSpam, len(vocab)])
	for i in range(numSpam):
		spamX[i] = readEmails("spam", i+1, vocab)
	spamY = np.ones([numSpam])

	easyHamX = np.zeros([numEasyHam, len(vocab)])
	for i in range(numEasyHam):
		easyHamX[i] = readEmails("easy_ham", i+1, vocab)
	easyHamY = np.zeros([numEasyHam])

	hardHamX = np.zeros([numHardHam, len(vocab)])
	for i in range(numHardHam):
		hardHamX[i] = readEmails("hard_ham", i+1, vocab)
	hardHamY = np.zeros([numHardHam])

	#X = np.vstack((spamX, easyHamX, hardHamX))
	#Y = np.vstack((spamY, easyHamY, hardHamY))

	print("creating splits")

	trainX = np.vstack((spamX[: int(0.6*np.shape(spamX)[0])], easyHamX[:int( 0.6 *
	                   np.shape(easyHamX)[0])], hardHamX[: int(0.6*np.shape(hardHamX)[0])]))

	testX = np.vstack((spamX[int(0.6 * np.shape(spamX)[0]): int(0.8 * np.shape(spamX)[0])], easyHamX[int(0.6 *
						np.shape(spamX)[0]): int(0.8 * np.shape(spamX)[0])],  hardHamX[int(0.6 * np.shape(spamX)[0]): int(0.8 * np.shape(spamX)[0])]))

	validateX=np.vstack((spamX[int(0.8*np.shape(spamX)[0]):], easyHamX[int(0.8 *
	                    np.shape(easyHamX)[0]):], hardHamX[int(0.8*np.shape(hardHamX)[0]):]))

	trainY =  np.hstack((spamY[:int(0.6*np.shape(spamY)[0])], easyHamY[:int(0.6 *
	                   np.shape(easyHamY)[0])], hardHamY[:int(0.6*np.shape(hardHamY)[0])]))

	testY = np.hstack((spamY[int(0.6 * np.shape(spamY)[0]): int(0.8 * np.shape(spamY)[0])], easyHamY[ int(0.6 *
						np.shape(spamY)[0]): int(0.8 * np.shape(spamY)[0])],  hardHamY[int(0.6 * np.shape(spamY)[0]):  int(0.8 * np.shape(spamY)[0])]))

	validateY = np.hstack((spamY[int(0.8*np.shape(spamY)[0]):], easyHamY[int(0.8 *
	                   np.shape(easyHamY)[0]):], hardHamY[int(0.8*np.shape(hardHamY)[0]):]))

	print("starting training")

	trainValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	svmOpt, cOpt,scoreOpt = entrenamiento(trainX, trainY, validateX, validateY, trainValues)

	testScore = svmOpt.score(testX,testY)
	print(f"Optimum C: {cOpt}")
	print(f"Error: {1-scoreOpt}")
	print(f"Precision: {testScore*100}%")




parte2()
