"""

This script define the different symbolic regularisation and sparsity function

"""
import theano
import theano.tensor as T

def l1_reg(x):
    return T.sum(T.abs_(x))

def l2_reg(x):
    return T.sum(x*x)

def l1_spar(x):
    return T.mean(T.sum(T.abs_(x),1),0)

def l2_spar(x):
    return T.mean(T.sum(x*x,1),0)
