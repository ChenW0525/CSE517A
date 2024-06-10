
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
    diff = np.dot(w.T, xTr) - yTr
    loss = np.dot(diff, diff.T) + lambdaa * np.dot(w.T, w)
    #----0129----- 1/n* square loss or just square loss?
    gradient = 2 * np.dot(xTr, diff.T) + 2 * lambdaa * w
    return loss, gradient
