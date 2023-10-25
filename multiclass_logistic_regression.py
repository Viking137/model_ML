import numpy as np
import sympy as sp
import math
from matplotlib import pyplot as plt


weight2 = np.array([1., 1., 1.])
weight1 = np.array([1., 1., 1.])
weight0 = np.array([1., 1., 1.])
features = np.array([ [1, -15, 5], [1, -12, 4], [1, -10, 10], [1, -7, 3], [1, -18, 9], [1, -14, 12], [1, -7, 6], [1, -9, 7], [1, -13, 4], [1, -5, 4],     
                        [1, 5, 1], [1, 10, -2], [1, 6, 8], [1, 9, 5], [1, 15, -3], [1, 13, 10], [1, 7, 12], [1, 4, 9], [1, 14, -3], [1, 11, 8],    
                        [1, -3, -5], [1, -2, -10], [1, -5, -8], [1, -7, -12], [1, -1, -14], [1, -9, -14], [1, 4, -10], [1, 9, -13], [1, 1, -6], [1, 11, -15]])   #train data
y = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

coef2, coef1, coef0 = 0.0008, 0.0008, 0.0008

for i in range(6000):  
    grad2, grad1, grad0 = np.array([0., 0., 0.]), np.array([0., 0., 0.]), np.array([0., 0., 0.])
    train_y = np.copy(y)  
    train_y[(train_y == 1 ) | (train_y == 0 )] = 0
    train_y[(train_y == 2 )] = 1
    for elem, yi in zip(features, train_y):
        p = 1 / (1 + math.exp(-yi)) 
        grad2 += elem * (yi - p)

    grad2 *= -(1 / features.shape[0])
    weight2 -= coef2*grad2

    train_y = np.copy(y) 
    train_y[(train_y == 2 ) | (train_y == 0 )] = 0
    train_y[(train_y == 1 )] = 1
    for elem, yi in zip(features, train_y):
        p = 1 / (1 + math.exp(-yi)) 
        grad1 += elem * (yi - p)

    grad1 *= -(1 / features.shape[0])
    weight1 -= coef1*grad1

    train_y = np.copy(y)  
    train_y[(train_y == 1 ) | (train_y == 2 )] = 0
    train_y[(train_y == 0 )] = 1
    for elem, yi in zip(features, train_y):
        p = 1 / (1 + math.exp(-yi)) 
        grad0 += elem * (yi - p)

    grad0 *= -(1 / features.shape[0])
    weight0 -= coef0*grad0
   
x0 = [ -15, -12, -10, -7, -18, -14, -7, -9, -13, -5]   
x1 = [ 5, 10, 6, 9, 15, 13, 7, 4, 14, 11]
x2 = [-3, -2, -5, -7, -1, -9, 4, 9, 1, 11]
y0 = [5, 4, 10, 3, 9, 12, 6, 7, 4, 4]
y1 = [ 1, -2, 8, 5, -3, 10, 12, 9, -3, 8]
y2 = [ -5, -10, -8, -12, -14, -14, -10, -13, -6, -15]
x = np.linspace(-16, 16, 100)
y_0 = -weight0[0]/weight0[2] - weight0[1]/weight0[2]*x
y_1 = -weight1[0]/weight1[2] - weight1[1]/weight1[2]*x
y_2 = -weight2[0]/weight2[2] - weight2[1]/weight2[2]*x

plt.scatter(x0,y0)
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.plot(x,y_0, '-', x,y_1, '--', x,y_2, ':') 
plt.show()

data_check = np.array([[1, -9, 3.5], [1, -3.5, 3], [1, 1.5, 1.5], [1, 6, -1], [1, -3.5, -4], [1, 1, -5]])

for elem in data_check:
    yi = np.dot(elem, weight0)
    p0 = 1 / (1 + math.exp(-yi))
    yi = np.dot(elem, weight1)
    p1 = 1 / (1 + math.exp(-yi))
    yi = np.dot(elem, weight2)
    p2 = 1 / (1 + math.exp(-yi))
    
    
    probability_ = [p2,p1,p0]
    #mega_exp = math.exp(p2) + math.exp(p1) + math.exp(p0)
    #map(lambda num: math.exp(num) / mega_exp,probability_)
    #print(probability_.index(max(probability_)))


