def count_line(a:float,b:float,X:list,line:list) -> list:
    """
    cout list of Y for given a and b
    """
    line.clear()
    for i in X:
        line.append(a*i+b)



X = [3.5, 0.5, 2.4, 2.8, 1.9, 3.8, 4.7, 5.1, 0.8, 4.9]
Y = [4.1, 0.2, 1.7, 2.5, 2.3, 3.6, 4.2, 5.4, 0.5, 5.2] 
line = []

b = 0
for i in Y:
    b += i
b /= len(Y)
k = 0
alpha = 0.005

grad_k = 0
grad_b = 0

i = 0
while i<30:
    for j in range(len(X)):
        grad_k += (2*k*X[j]+2*(b-Y[j]))*X[j]
        grad_b += 2*k*X[j] + 2*b - 2*Y[j]

    grad_b /= len(X)
    grad_k /= len(X)


    
    k -= alpha*grad_k
    b -= alpha*grad_b

    grad_b = 0
    grad_k = 0

   # count_line(k,b,X,line)
    print(k,b)
   # print(line)
    print("----------------------------------")
    i += 1