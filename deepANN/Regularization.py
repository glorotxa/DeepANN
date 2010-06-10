"""

This script define the different symbolic regularisation and sparsity function

"""
import theano
import theano.tensor as T

def l1(x):
    return T.mean(T.sum(T.abs_(x),1),0)

def l2(x):
    return T.mean(T.sum(x*x,1),0)

def l1_target(t):
    def real_func(x):
        return T.mean(T.sum(T.abs_(x-t),1),0)
    return real_func

def l2_target(t):
    def real_func(x):
        return T.mean(T.sum((x-t)*(x-t),1),0)
    return real_func
