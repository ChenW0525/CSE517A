#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: Yichen
@author: M.Joo (smoothing with all zeros)
"""

import numpy as np

def naivebayesPXY(x, y):
# =============================================================================
#    function [posprob,negprob] = naivebayesPXY(x,y);
#
#    Computation of P(X|Y)
#    Input:
#    x : n input vectors of d dimensions (dxn)
#    y : n labels (-1 or +1) (1xn)
#
#    Output:
#    posprob: dx1 probability vector with entries p(x_alpha = 1|y=+1)
#    negprob: dx1 probability vector with entries p(x_alpha = 1|y=-1)
# =============================================================================



    # Convertng input matrix x and y into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    # TODO: do not use np.matrix!
    X = np.matrix(x)
    Y = np.matrix(y)

    d,n = X.shape

    # Pre-constructing a matrix of all-ones (dx2)
    X0 = np.ones((d,2))
    Y0 = np.array([[-1, 1]])

    # add one all-ones positive and negative example
    Xnew = np.hstack((X, X0)) #stack arrays in sequence horizontally (column-wise)
    Ynew = np.hstack((Y, Y0))

    # matrix of all-zeros -
    X1 = np.zeros((d, 2))
    # add one all-zeros positive and negative example - M.Joo
    Xnew = np.hstack((Xnew, X1))
    Ynew = np.hstack((Ynew, Y0))

    # Re-configuring the size of matrix Xnew
    d,n = Xnew.shape

# =============================================================================
# fill in code here
    # YOUR CODE HERE
    # posprob = np.zeros((d, 1))
    # negprob = np.zeros((d, 1))
    # posInd = np.where(y == 1)[0]
    # negInd = np.where(y == -1)[0]
    # for i in range(d):
    #     posNum = np.sum(Xnew[i, posInd])
    #     negNum = np.sum(Xnew[i, negInd])
    #     posprob[i] = posNum / len(posInd)
    #     negprob[i] = negNum / len(negInd)
    posMask = Ynew == 1
    negMask = Ynew == -1
    posMat = posMask * Xnew.T
    negMat = negMask * Xnew.T
    # posNum = np.sum(posMat, axis=1)
    # negNum = np.sum(negMat, axis=1)
    # posCnt = np.sum(posMask)
    # posCnt2 = np.sum(np.where(y == 1))
    posprob = posMat / np.sum(posMask)# ___0213____Divider may wrong
    negprob = negMat / np.sum(negMask)

    return posprob.T,negprob.T

# =============================================================================
