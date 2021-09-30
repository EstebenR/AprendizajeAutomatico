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
    mat_norm = np.empty_like(mat)
    medias = np.empty(mat.shape[1])
    desviaciones = np.empty(mat.shape[1])
    for i in range(mat.shape[1]):
        col = mat[:,i]
        #mat_norm[:,i] = col/np.linalg.norm(col)
        medias[i] = np.mean(col)
        desviaciones[i] = np.std(col)
        mat_norm[:,i] = (col-medias[i])/desviaciones[i]
        #vectorizar pidiendo media por columnas
    return mat_norm, medias, desviaciones


def regresion_varias():
    datos = carga_csv("ex1data2.csv")
    #print(datos)
    mat_norm, media, desv = normaliza_matriz(datos)
    print(mat_norm)
    print(media)


#regresion_lineal()
regresion_varias()