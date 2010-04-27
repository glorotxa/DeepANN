"""

This script define the different symbolic activations function

"""
import theano
import theano.tensor as T

def sigmoid_act(x):
    return theano.tensor.nnet.sigmoid(x)

def tanh_act(x):
    return theano.tensor.tanh(x)

def tanhnorm_act(x): #@TO DO:cross-entropy pending
    return 1.759*theano.tensor.tanh(0.6666*x)

def abstanh_act(): #@TO DO:cross-entropy pending
    return theano.tensor.abs_(theano.tensor.tanh(x))

def abstanhnorm_act(): #@TO DO:cross-entropy pending
    return theano.tensor.abs_(1.759*theano.tensor.tanh(0.6666*x))

def softsign_act(x):
    return x/(1.0 + T.abs_(x))

def abssoftsign_act():
    return T.abs_(x)/(1.0 + T.abs_(x))

def rectifier_act(x):
    return x*(x>=0)

def softplus_act(x): #@TO DO: rescale in order to have a steady state regime close to 0 at init.
    return T.log(1+T.exp(x))

# @TO TEST---------------------------------------------------------------------
def arsinh_act(x):
    return T.log(x+T.sqrt(1+x*x))

def pl_act(x): #harder to optimize (2 plateaus instead of 1)
    return x*(x>=0)*(x<=1)+T.ones_like(x)*(x>1)

def plc_act(x): #imposed symmetry not good
    return -T.ones_like(x)*(x<-1)+x*(x>=-1)*(x<=1)+T.ones_like(x)*(x>1)

def pascalv_act(x): 
    return T.log(1+T.abs_(x))

def neuronlike_act(x):#@TO DO
    pass


#----------------------------------------------------------------- derivated function


def sigmoid_der(x):
    y = theano.tensor.nnet.sigmoid(x)
    return y * (1-y)

def tanh_der(x):
    y = theano.tensor.tanh(x)
    return 1 - y*y

def tanhnorm_der(x): #@TO DO:cross-entropy pending
    y = theano.tensor.tanh(x)
    return 1.759 * 0.6666 * (1-theano.tensor.tanh(0.6666*x)*theano.tensor.tanh(0.6666*x))

def abstanh_der(x): #@TO DO:cross-entropy pending
    y = theano.tensor.tanh(x)
    return T.sgn(y) * (1 - y*y)

def abstanhnorm_der(): #@TO DO:cross-entropy pending
    y = theano.tensor.tanh(0.6666*x)
    return T.sgn(y) * 1.759 * 0.6666 * (1-y*y)

def softsign_der(x):
    return 1/((1+T.sgn(x)*x) * (1+T.sgn(x)*x))

def abssoftsign_der(x):
    return T.sgn(x)/((1+T.sgn(x)*x) * (1+T.sgn(x)*x))

def rectifier_der(x):
    return 1.*(x>=0)

def softplus_der(x): #@TO DO: rescale in order to have a steady state regime close to 0 at init.
    return theano.tensor.nnet.sigmoid(x)

# @TO TEST---------------------------------------------------------------------
def arsinh_der(x):
    y = T.sqrt(1+x*x)
    return (1.0 + x/y) / (x + y)

def pl_der(x): #harder to optimize (2 plateaus instead of 1)
    return 1.*(x>=0)*(x<=1)

def plc_der(x): #imposed symmetry not good
    return 1.0*(x>=-1)*(x<=1)

def pascalv_der(x):
    return (1.0 + T.sgn(x)) / (x + T.abs_(x))

def neuronlike_der(x):#@TO DO
    pass

