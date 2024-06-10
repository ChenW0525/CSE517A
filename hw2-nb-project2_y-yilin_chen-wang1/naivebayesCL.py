#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
@author: MN (categorical/Bernoulli NB)
"""

import numpy as np
from naivebayesPY import naivebayesPY
from naivebayesPXY import naivebayesPXY

def naivebayesCL(x, y):
# =============================================================================
#function [w,b]=naivebayesCL(x,y);
#
#Implementation of a Naive Bayes classifier
#Input:
#x : n input vectors of d dimensions (dxn)
#y : n labels (-1 or +1)
#
#Output:
#w : weight vector
#b : bias (scalar)
# =============================================================================



    # Convertng input matrix x and x1 into NumPy matrix
    # input x and y should be in the form: 'a b c d...; e f g h...; i j k l...'
    X = np.matrix(x)

    # Pre-configuring the size of matrix X
    d,n = X.shape

# =============================================================================
# fill in code here
    # YOUR CODE HERE
    # w = np.zeros((d, 1))
    pos, neg = naivebayesPY(x, y)
    posProb, negProb = naivebayesPXY(x, y)
    w = np.log(posProb) - np.log(negProb)
    b = np.log(pos) - np.log(neg)
    # w = np.log(posProb / (1 - posProb)) - np.log(negProb / (1 - negProb))
    # b = np.log(pos) - np.log(neg) + np.log((1 - posProb) / (1 - negProb))

    return w,b
# =============================================================================
