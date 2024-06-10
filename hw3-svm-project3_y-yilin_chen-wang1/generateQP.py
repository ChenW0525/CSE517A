"""
INPUT:	
K : nxn kernel matrix
yTr : nx1 input labels
C : regularization constant

Output:
Q,p,G,h,A,b as defined in qpsolvers.solve_qp

A call of qpsolvers.solve_qp(Q, p, G, h, A, b) should return the optimal nx1 vector of alphas
of the SVM specified by K, yTr, C. Just make these variables np arrays.

"""
import numpy as np

def generateQP(K, yTr, C):
    yTr = yTr.astype(np.double)
    n = yTr.shape[0]
    
    # YOUR CODE HERE
    Q = np.dot(yTr, yTr.T) * K
    p = np.ones((n, 1)) * (-1)

    id_mat = np.eye(n)
    G = np.vstack((id_mat, id_mat * (-1)))
    c_mat = np.ones((n, 1)) * C
    zero_mat = np.zeros((n, 1))
    h = np.vstack((c_mat, zero_mat))

    A = np.array(yTr.T)
    b = np.zeros((1, 1))
            
    return Q, p, G, h, A, b

