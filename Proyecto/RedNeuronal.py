import numpy as np
from pandas.io.parsers import read_csv
from scipy import optimize

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

def sigmoide(x):
	return 1/( 1 + np.exp(-x)) 

def propagacion(X,theta1,theta2):
	m = np.shape(X)[0]
	A1 = np.hstack([np.ones([m,1]),X])
	Z2 = np.dot(A1, theta1.T)
	A2 = np.hstack([np.ones([m,1]),sigmoide(Z2)])
	Z3 = np.dot(A2, theta2.T)
	A3 = sigmoide(Z3)
	return A1, A2, A3

def costeSinRegularizar(X,y, t1,t2):
	m = np.shape(X)[0]
	a1,a2,h = propagacion(X,t1,t2)
	part1 = y*np.log(h)
	part2 = (1-y)*np.log(1-h+1e-9)
	return (-1/m)*np.sum(part1+part2)

def costeRegularizado(X,y, t1,t2, reg):
	m = np.shape(X)[0]
	
	aux = costeSinRegularizar(X, y, t1, t2)
	otra = (reg/(2*m)) * (np.sum(np.power(t1[1:],2)) +np.sum(np.power(t2[1:], 2)))
	
	return aux + otra

def terminoRegularizacion(gradiente, m, reg, theta):
	columnaGuardada = gradiente[0]
	gradiente = gradiente + (reg/m)*theta
	gradiente[0] = columnaGuardada
	return gradiente

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	m = np.shape(X)[0]

	Theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)] , (num_ocultas, (num_entradas+1)))
	Theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):] , (num_etiquetas, (num_ocultas+1)))

	Delta1 = np.zeros_like(Theta1)
	Delta2 = np.zeros_like(Theta2)

	A1, A2, H = propagacion(X,Theta1, Theta2)

	for t in range(m):
		a1t = A1[t,:]
		a2t = A2[t,:]
		ht = H[t,:]
		yt = y[t]

		d3t = ht -yt
		d2t = np.dot(Theta2.T, d3t)*(a2t*(1-a2t))

		Delta1 = Delta1 + np.dot(d2t[1:,np.newaxis], a1t[np.newaxis,:])
		Delta2 = Delta2 + np.dot(d3t[:,np.newaxis],a2t[np.newaxis,:])

	gradiente1 = Delta1/m
	gradiente2 = Delta2/m

	gradiente1 = terminoRegularizacion(gradiente1, m, reg,Theta1)
	gradiente2 = terminoRegularizacion(gradiente2, m, reg,Theta2)

	return costeRegularizado(X,y,Theta1,Theta2,reg), np.concatenate([np.ravel(gradiente1),np.ravel(gradiente2)])

def main():
	X, Y = cargaDatos()
	
	num_ocultas = 25
	num_etiquetas = 7
	num_entradas = np.shape(X)[1]
	
	epsilon = 0.12
	reg = 1
	nIters = 100
	pesos = np.random.uniform(-epsilon,epsilon, num_ocultas*(num_entradas+1)+(num_ocultas+1)*num_etiquetas)
	res = optimize.minimize(fun=backprop, x0=pesos, args=(np.shape(X)[1],num_ocultas,num_etiquetas,X,Y,reg), jac = True, method = 'TNC', options={'maxiter':nIters})
	
	Theta1 = np.reshape(res.x[:num_ocultas * (num_entradas + 1)] , (num_ocultas, (num_entradas+1)))
	Theta2 = np.reshape(res.x[num_ocultas * (num_entradas + 1):] , (num_etiquetas, (num_ocultas+1)))
	
	alpha3 = propagacion(X,Theta1,Theta2)[2]

	maxIndices = np.argmax(alpha3,axis=1)
	maxIndices = maxIndices
	acertados = np.sum(maxIndices==np.where(Y[:] == 1))
	print(f"Porcentaje de valores que han sido correctamente clasificados: {acertados*100/np.shape(alpha3)[0]:.3f}%")

main()