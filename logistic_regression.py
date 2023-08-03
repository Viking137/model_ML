import numpy as np
import math
from matplotlib import pyplot as plt

weight = np.array([1., 1., 1.])
features = np.array([ [1, 2, 4], [1, 2, -2], [1, 4, 2], [1, 6, -2], [1, -2, 4], [1, -4, 6], [1, 14, -6], [1, 16, 2], [1, 8, 12], [1, -8, 12],    
                        [1, 2, -4], [1, -4, 2], [1, 4, -12], [1, -12, -12], [1, -10, -4], [1, -6, 2], [1, -14, 4], [1, -12, 6], [1, -6, 2], [1, -12, 8]])   #train data
y = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])

alpha = 0.00001
grad = np.array([0., 0., 0.])


for i in range(2000):    
    for elem, yi in zip(features, y):
        p = 1 / (1 + math.exp(-yi)) #probability
        grad += elem * (yi - p)

    grad *= -(1 / features.shape[0])

    weight -= alpha*grad

print(weight)

x1 = [2,2,4,6,-2,-4,14,16,8,-8]
x2 = [2,-4,4,-12,-10,-6,-14,-12,-6,-12]
y1 = [4,-2,2,-2,4,6,-6,2,12,12]
y2 = [-4,2,-12,-12,-4,2,4,6,2,8]
x = np.linspace(-10, 10, 100)
y = -weight[0]/weight[2] - weight[1]/weight[2]*x

plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.plot(x,y)
plt.show()
   
x_check = np.array([[1, 2, 2], [1, 4, -4], [1, -4, -4], [1, -8, 2]])

for elem in x_check:
    yi = np.dot(elem, weight)
    p = 1 / (1 + math.exp(-yi))
    if p > 0.5:
        print(1)
    elif p < 0.5:
        print(0)
    else:
        print("on line")

