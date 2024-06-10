import numpy as np

"""
function D=l2distance(X,Z)
	
Computes the Euclidean distance matrix. 
Syntax:
D=l2distance(X,Z)
Input:
X: dxn data matrix with n vectors (columns) of dimensionality d
Z: dxm data matrix with m vectors (columns) of dimensionality d

Output:
Matrix D of size nxm 
D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
"""

def l2distance(X,Z):
    d, n = X.shape
    dd, m = Z.shape
    assert d == dd, 'First dimension of X and Z must be equal in input to l2distance'
    
    D = np.zeros((n, m))
    
    # YOUR CODE HERE
    sq = np.sum((X[:, :, np.newaxis] - Z[:, np.newaxis, :]) ** 2, axis=0)
    D = np.sqrt(sq)
    
    return D

# d = 2
# vecs_1 = np.random.rand(d, 3)
# vecs_2 = np.random.rand(d, 4)
# dist = l2distance(vecs_1, vecs_2)
# print(dist)