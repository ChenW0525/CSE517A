import math
import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    pred = np.dot(xTr.T, w)
    power = (yTr.T) * pred
    loss = np.log(1 + np.exp(-power))
    yx = (yTr * xTr).T
    gradient = -yx / (1 + np.exp(power))

    loss = np.sum(loss, axis=0).reshape(1, 1)
    gradient = np.sum(gradient, axis=0).reshape(w.shape)
    return loss, gradient
