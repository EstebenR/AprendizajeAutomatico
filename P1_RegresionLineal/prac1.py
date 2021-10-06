import numpy as np
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def carga_csv(file_name):
    valores = read_csv(file_name,header=None).to_numpy()

    return valores.astype(float)

def coste(X,Y,theta_0,theta_1):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += ((theta_0+theta_1*X[i])-Y[i])**2
    return 1/(2*m)*sum

def regresion_lineal():
    datos = carga_csv("ex1data1.csv")
    X = datos[:,0]
    Y = datos[:,1]
    m = len(X)
    alpha = 0.01
    theta_0 = theta_1 = 0
    #print(coste(X,Y,theta_0,theta_1))
    for _ in range(1500):
        sum_0 = sum_1 = 0
        for i in range(m):
            sum_0 += (theta_0 + theta_1 * X[i]) - Y[i]
            sum_1 += ((theta_0 + theta_1 * X[i]) - Y[i]) * X[i]
        theta_0 = theta_0 - (alpha/m) * sum_0
        theta_1 = theta_1 - (alpha/m)*sum_1
        #print(coste(X,Y,theta_0,theta_1))
    make_data([-10,10],[-1,4],X,Y,[theta_0,theta_1])
    plt.plot(X,Y,"x")
    min_x = min(X)
    max_x = max(X)
    min_y = theta_0 + theta_1 * min_x
    max_y = theta_0 + theta_1 * max_x
    print(f"{theta_0} + {theta_1}*x")
    plt.plot([min_x, max_x], [min_y,max_y])
    plt.savefig("resultado.png")

def make_data(t0_range, t1_range, X,Y,min_point):
    step = 0.1
    theta_0 = np.arange(t0_range[0],t0_range[1], step)
    theta_1 = np.arange(t1_range[0],t1_range[1], step)
    theta_0,theta_1 = np.meshgrid(theta_0,theta_1)
    cost = np.empty_like(theta_0)
    for ix, iy in np.ndindex(theta_0.shape):
        cost[ix,iy] = coste(X,Y,theta_0[ix,iy],theta_1[ix,iy])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta_0,theta_1,cost,cmap=cm.jet)
    ax.set_xlabel("θ0")
    ax.set_ylabel("θ1")
    ax.set_zlabel("coste")
    plt.savefig("superficie.png")
    plt.clf()

    plt.plot(min_point[0],min_point[1],marker='x',markersize=10,markeredgecolor='red')
    plt.contour(theta_0,theta_1,cost,np.logspace(-2,3,20),colors='blue')
    plt.savefig("contorno.png")
    plt.clf()


def normaliza_matriz(mat):
    medias = mat.mean(axis=0)
    desviaciones = mat.std(axis=0)
    mat_norm = (mat[:]-medias)/desviaciones
    return mat_norm, medias, desviaciones

def coste_vectorizado(X, Y, Theta):
    Hipotesis = np.dot(X, Theta)
    Aux = (Hipotesis - Y) ** 2
    return Aux.sum() / (2 * len(X))

def regresion_varias():
    datos = carga_csv("ex1data2.csv")
    mat_norm, medias, desvs = normaliza_matriz(datos)
    X = mat_norm[:,:-1]
    Y = mat_norm[:,-1]
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m,1]),X])

    plt.figure()

    alpha = 0.01
    Thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='orange',label='alfa 0.01')
    alpha = 0.03
    Thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='green',label='alfa 0.03')
    alpha = 0.1
    Thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='blue',label='alfa 0.1')
    alpha = 0.3
    Thetas, costes = descenso_gradiente(X,Y,alpha)
    plt.scatter(np.arange(np.shape(costes)[0]),costes,c='red',label='alfa 0.3')
    
    plt.legend()
    plt.savefig("Decenso gradiente")

    #Probamos a predecir el precio de una casa de 1650 pies cuadrados y 3 habitaciones
    #Primero normalizamos los valores para meterlos a la funcion
    pies = (1650 - medias[0])/desvs[0]
    habs = (3 - medias[1])/desvs[1]
    precio = (Thetas[0]+Thetas[1]*pies+Thetas[2]*habs)*desvs[2]+medias[2]
    print(f"Precio predecido para 1650 pies con 3 habitaciones con descenso de gradiente: {precio}")

def descenso_gradiente(X, Y, alpha):
    Theta = np.zeros(np.shape(X)[1])
    num_intentos = 300
    costes = np.zeros(num_intentos)
    for i in range(num_intentos):
        costes[i] = coste_vectorizado(X,Y,Theta)
        Aux = gradiente2(X,Y,Theta,alpha)
        Theta = Aux

    return Theta,costes

def gradiente1(X,Y,Theta, alpha):
    NuevaTheta = Theta
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    H = np.dot(X, Theta)
    Aux = (H - Y)
    for i in range(n):
        Aux_i = Aux * X[:, i]
        NuevaTheta[i] -= (alpha / m) * Aux_i.sum()
    return NuevaTheta

def gradiente2(X,Y,Theta, alpha):
    m = np.shape(X)[0]
    H = np.dot(X, Theta)
    return Theta - (alpha/m) * np.dot(np.transpose(X), (H-Y))

def ec_normal(X,Y):
    return np.dot(np.linalg.pinv(np.dot(np.transpose(X), X)),np.dot(np.transpose(X), Y))

def resultado_normal():
    datos = carga_csv("ex1data2.csv")
    X = datos[:,:-1]
    Y = datos[:,-1]
    m = np.shape(X)[0]
    X = np.hstack([np.ones([m,1]),X])
    Thetas = ec_normal(X,Y)
    pies = 1650
    habs = 3
    precio = (Thetas[0]+Thetas[1]*pies+Thetas[2]*habs)
    print(f"Precio predecido para 1650 pies con 3 habitaciones con ecuacion normal: {precio}")



#regresion_lineal()
regresion_varias()
resultado_normal()