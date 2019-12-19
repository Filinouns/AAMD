import numpy as np
import matplotlib.pyplot as plt
from scipy . io import loadmat
import scipy.optimize as opt

data = loadmat('ex3data1.mat')
# se pueden consultar las claves con data . keys() 5 
y = data ['y']
X = data ['X']
# almacena los datos leídos en X, y

m = X.shape[0]
n = X.shape[1]

XStack = np.hstack([np.ones((len(X), 1)), X])
y = np.ravel(y)

#print(XStack.shape)

# Selecciona aleatoriamente 10 ejemplos y los pinta
sample = np.random.choice(m , 10) 
plt.imshow(X[sample, :].reshape(-1, 20).T)
#plt.axis('on')