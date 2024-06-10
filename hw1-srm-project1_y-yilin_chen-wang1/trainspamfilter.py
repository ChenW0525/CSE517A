
import numpy as np
from ridge import ridge
from hinge import hinge
from logistic import logistic
from grdscent import grdescent
from scipy import io
from checkgradHingeAndRidge import checkgradHingeAndRidge

def trainspamfilter(xTr,yTr):
    #
    # INPUT:
    # xTr
    # yTr
    #
    # OUTPUT: w_trained
    #
    # Consider optimizing the input parameters for your loss and GD!


    # (not the most successful) EXAMPLE:
    # print("...using ridge")
    # f = lambda w : ridge(w,xTr,yTr,1e-02)

    # print("...using hinge")
    # f = lambda w : hinge(w, xTr, yTr, 1)

    print("...using logistic")
    f = lambda w : logistic(w, xTr, yTr)

    # normDiff = checkgradHingeAndRidge(f, np.zeros((xTr.shape[0],1)), 1e-02, xTr, yTr, 1)
    # print("check ridge: ", normDiff)
    w_trained = grdescent(f,np.zeros((xTr.shape[0],1)),1e-04,1000)

    # YOUR CODE HERE


    io.savemat('w_trained.mat', mdict={'w': w_trained})
    return w_trained
