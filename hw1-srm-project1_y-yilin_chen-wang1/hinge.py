from numpy import maximum
import numpy as np


def hinge(w,xTr,yTr,lambdaa):
#
#
# INPUT:
# xTr dxn matrix (each column is an input vector)
# yTr 1xn matrix (each entry is a label)
# lambda: regularization constant
# w weight vector (default w=0)
#
# OUTPUTS:
#
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w


    # YOUR CODE HERE
    pred = np.dot(xTr.T, w)
    y = yTr.T
    predY = y * pred
    yx = - xTr.T * y
    # yx = np.sum(yx, axis=1).reshape(y.shape)
    predY = np.ones(y.shape) - predY
    # n = np.shape(w)[0]
    loss = np.sum(np.maximum(predY, np.zeros(y.shape))) + lambdaa * np.dot(w.T, w)
    mask = predY > 0
    gradient = np.where(mask, yx, np.zeros(yx.shape))
    gradient = np.sum(gradient.T, axis=1).reshape(w.shape) + 2 * lambdaa * w
    # loss = lambdaa * np.dot(w.T, w)
    # gradient = 2 * lambdaa * w
    # if predY < 1:
    #     loss = loss + 1 - predY
    #     gradient = gradient - np.dot(yTr, xTr.T).reshape(w.shape)
    return loss, gradient
