from re import X
import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from sklearn.preprocessing import PolynomialFeatures

categories = {"RP" , "EC" , "E", "ET" , "T" , "M" , "A"}

def cargaDatos():
	table = read_csv('Video_games_esrb_rating.csv',header=0,).to_numpy()
	X = table[:,1:-1]
	#Hay 7 tipos de ratings que vienen codificados con un string, asi que los codificamos como ints
	Y = np.empty([np.shape(table)[0],7])
	print(np.shape(Y))
	print(table[:,-1])
	for index, cat in enumerate(categories):
		col = table[:,-1]==cat
		Y[:,index] = col
		
	return X.astype(int),Y.astype(int)

x,y = cargaDatos()

print(y)