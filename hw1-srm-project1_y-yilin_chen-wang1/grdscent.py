
import numpy as np
def grdescent(func,w0,stepsize,maxiter,tolerance=1e-02):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent

    # YOUR CODE HERE
    gradIter = 0
    w = w0
    stepS = stepsize
    while gradIter < maxiter:
        fx, dfx = func(w)
        gradNorm = np.linalg.norm(dfx)
        if gradNorm < tolerance:
            break
        newW = w - stepS * dfx
        newFx, newDfx = func(newW)
        if newFx < fx:
            stepS = stepS * 1.01
        else:
            #--------0130--------use dfx or newDfx in line 30?
            newW = w + stepS * dfx
            if stepS < eps:
                stepS = eps
            else: 
                stepS = stepS * 0.5
        gradIter += 1
        w = newW
    return w
