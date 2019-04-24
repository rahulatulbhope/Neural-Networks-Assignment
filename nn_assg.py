import numpy as np
from numpy import genfromtxt

from sklearn.metrics import accuracy_score

# input array
X = genfromtxt('train_x_labels.csv',delimiter=',')  # train data

X[0][0] = 0

# output array
Y = genfromtxt('train_y_labels.csv',delimiter=',')  # train data
Y = np.reshape(Y,(1,22559))

Y[0][0] = -1

test = genfromtxt('test_x_labels.csv',delimiter=',')
print(X.shape)
print(Y.shape)
print(test.shape)


X_flat = X.T
W = np.random.rand(12, 1) * 0.01
b = 0
m = Y.size
print(np.shape(X))
print(np.shape(Y))

# Activation Function

def activation_function(z):
    s = np.tanh(z)
    # s = 1 / (1 + (np.exp(-z)))
    # leak = 0.2
    # f1 = 0.5 * (1 + leak)
    # f2 = 0.5 * (1 - leak)
    # s = f1 * z + f2 * abs(z)
    return s




epoch = 2000  # Setting training iterations
lr = 0.7  # Setting learning rate
print("Training.....")
for i in range(epoch):
    # Forward Propogation
    Z = np.dot(W.T, X_flat) + b
    A = activation_function(Z)
    #print(A)
    cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1, keepdims=True)
    # Backpropagation
    dZ = Y - A
    dw = 1 / m * np.dot(X_flat, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    W = W - lr * dw
    b = b - lr * db

vali = np.array(activation_function(np.dot(W.T, X_flat) + b))


#print(vali)
#print(vali.shape)

print(vali.shape)

vali = np.reshape(vali,(1,22559))

a_vali = np.where(vali > 0,1,-1)

Y_true = a_vali.flatten()
Y_pred = Y.flatten()
print("Accuracy is .........")
accuracy = (np.abs(Y_true - Y_pred) < 0 ).all(axis=(0)).mean()
print(accuracy_score(Y_true,Y_pred) * 100)


#accu = np.asarray([vali == y for y in Y])

#accu = np.reshape(accu,(1,22559))
#print(accu.shape)

#print(np.sum(accu))



M = test  # test data
M [0][0] = 0
M_flat = M.T
#print(W, b)
out = np.array(activation_function(np.dot(W.T, M_flat) + b))
out_1 = np.where(out > 0,1,-1)
out_1 = out_1.T
np.savetxt("foo_3.csv", out_1, delimiter=",")

#for d in np.nditer(out, op_flags=['readwrite']):
#    print(d)

#print(np.shape(out))
#print(np.shape(Y))
