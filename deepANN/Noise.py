"""

This script define the different symbolic noise functions
The noise contract is simple: noise_lvl is a symbolic variable going from 0 to 1
0: no changement.
1: max noise.

"""
import theano
import theano.tensor as T

def binomial_noise(theano_rng,inp,noise_lvl):
    return theano_rng.binomial( size = inp.shape, n = 1, prob =  1 - noise_lvl) * inp
