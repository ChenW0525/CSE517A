"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
alphas  : nx1 vector or alpha values
C : regularization constant

Output:
bias : the scalar hyperplane bias of the kernel SVM specified by alphas

Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
0<alpha<C
"""

import numpy as np

def recoverBias(K,yTr,alphas,C):
    bias = 0
    n = yTr.shape[0]
    # YOUR CODE HERE
    dist = np.abs(alphas - C / 2)
    i = np.argmin(dist)
    k_i = K[i, :].reshape(-1, 1)
    bias = yTr[i] - np.dot(alphas.T, (yTr * k_i))
    
    return bias 
    
