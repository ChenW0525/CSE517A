"""
INPUT:	
xTr : dxn input vectors
yTr : 1xn input labels
ktype : (linear, rbf, polynomial)
Cs   : interval of regularization constant that should be tried out
paras: interval of kernel parameters that should be tried out

Output:
bestC: best performing constant C
bestP: best performing kernel parameter
lowest_error: best performing validation error
errors: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)

Trains an SVM classifier for all combination of Cs and paras passed and identifies the best setting.
This can be implemented in many ways and will not be tested by the autograder. You should use this
to choose good parameters for the autograder performance test on test data. 
"""

import numpy as np
import math
from trainsvm import trainsvm
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization
from functools import partial

def trainsvm_error(xTr, yTr, ktype, C, P):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    err_list = []
    # mat_p = np.full((1, 1), P)
    for train_i, val_i in kf.split(yTr):
        x_trainf, x_valf = (xTr.T[train_i]).T, (xTr.T[val_i]).T
        y_trainf, y_valf = yTr[train_i], yTr[val_i]
        clf = trainsvm(x_trainf, y_trainf, C, ktype, P)
        y_pred = clf(x_valf)
        train_err = np.mean(y_pred != y_valf)
        err_list.append(train_err)
    avg_err = sum(err_list) / len(err_list)
    return -avg_err

def crossvalidate(xTr, yTr, ktype, Cs, paras):
    bestC, bestP, lowest_error = 0, 0, 0
    errors = np.zeros((len(paras),len(Cs)))
    
    # YOUR CODE HERE
    pbounds = {
        'C': (0.1, 10), 
        # 'ktype': (['linera', 'poly', 'rbf']), 
        'P': (1, 10)
    }
    f_partial = partial(trainsvm_error, xTr=xTr, yTr=yTr, ktype=ktype)
    optimizer = BayesianOptimization(
        f = f_partial,
        pbounds = pbounds, 
        verbose = 2
    )
    optimizer.maximize(init_points=5, n_iter=10)
    best_params = optimizer.max['params']
    bestC = best_params['C']
    bestP = best_params['P']
    lowest_error = -optimizer.max['target']
    return bestC, bestP, lowest_error, errors


    