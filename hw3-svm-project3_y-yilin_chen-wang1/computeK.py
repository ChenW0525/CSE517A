"""
function K = computeK(kernel_type, X, Z)
computes a matrix K such that Kij=g(x,z);
for three different function linear, rbf or polynomial.

Input:
kernel_type: either 'linear','poly','rbf'
X: n input vectors of dimension d (dxn);
Z: m input vectors of dimension d (dxm);
kpar: kernel parameter (inverse kernel width gamma in case of RBF, degree in case of polynomial)

OUTPUT:
K : nxm kernel matrix
"""
import numpy as np
from l2distance import l2distance

def computeK(kernel_type, X, Z, kpar):
    assert kernel_type in ['linear', 'poly', 'rbf'], kernel_type + ' is an unrecognized kernel type in computeK'
    
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to computeK'
    
    K = np.zeros((n,m))
    
    # YOUR CODE HERE
    if kernel_type == 'linear':
        K = np.dot(X.T, Z)
    elif kernel_type == 'poly':
        mat_fac = np.dot(X.T, Z) + np.ones((n, m))
        K = mat_fac ** kpar
    elif kernel_type == 'rbf':
        mat_dist = l2distance(X, Z) ** 2
        K = np.exp(-kpar * mat_dist)
    
    return K
