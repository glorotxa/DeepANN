"""
This tutorial introduces logistic regression using Theano and stochastic 
gradient descent.  

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability. 

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of 
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method 
suitable for large datasets, and a conjugate gradient optimization method 
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" - 
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import numpy, time, cPickle, gzip

import sys
import theano
import theano.tensor as T

from Regularization import *


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W` 
    and bias vector :math:`b`. Classification is done by projecting data 
    points onto a set of hyperplanes, the distance to which is used to 
    determine a class membership probability. 
    """
    def __init__(self,rng, inp, n_inp, n_out, wdreg = 'l2',upmaskbool = False, Winit = None, binit = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the 
                      architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in 
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in 
                      which the labels lie

        """ 
        self.inp = inp
        self.n_out = n_out
        self.n_inp = n_inp
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        allocW = False
        if Winit == None:
            wbound = numpy.sqrt(6./(n_inp+n_out))
            W_values = numpy.asarray( rng.uniform( low = -wbound, high = wbound, \
                                        size = (n_inp, n_out)), dtype = theano.config.floatX)
            self.W = theano.shared(value = W_values, name = 'W')
            allocW = True
        else:
            self.W = Winit
        
        # initialize the baises b as a vector of n_out 0s
        allocb = False
        if binit == None:
            self.b = theano.shared(value=numpy.zeros((n_out,), dtype = theano.config.floatX),\
                                                    name='b')
            allocb = True
        else:
            self.b = binit
        self.lin = T.dot(self.inp, self.W)+self.b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(self.lin)
        self.out = self.p_y_given_x

        # compute prediction as class whose probability is maximal in 
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)
        self.wdreg = wdreg
        self.wdreg_fn = eval(wdreg)
        # parameters of the model
        self.params = [self.W, self.b]
        self.wd = self.wdreg_fn(self.W)
        self.upmaskbool = upmaskbool
        if self.upmaskbool:
            self.updatemaskinit()
        print >> sys.stderr, '\t\t**** Logistic_regression.__init__ ****'
        print >> sys.stderr, '\t\tinp = ', inp
        print >> sys.stderr, '\t\tn_inp = ', self.n_inp
        print >> sys.stderr, '\t\tn_out = ', self.n_out
        print >> sys.stderr, '\t\tout = ', self.out
        print >> sys.stderr, '\t\tparams (gradients) = ', self.params
        print >> sys.stderr, '\t\twdreg (weigth decay) = ', self.wdreg
        print >> sys.stderr, '\t\tupmaskbool = ', self.upmaskbool
        print >> sys.stderr, '\t\tallocW, allocb = ', allocW, allocb
    
    def cost(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e., number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1]
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP) with one row per example and one column per class 
        # LP[T.arange(y.shape[0]),y] is a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]]
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch 
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the 
                  correct label
        """

        # check if y has same dimension of y_pred 
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype        
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.cast(T.mean(T.neq(self.y_pred, y)),theano.config.floatX)
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
    def updatemaskinit(self):
        self.W_upmask = theano.shared(value= numpy.ones((self.n_inp, self.n_out),\
                        dtype= theano.config.floatX), name = 'W_fact')
        self.b_upmask = theano.shared(value= numpy.ones((self.n_out,),\
                        dtype= theano.config.floatX), name = 'b_fact')
        self.upmask = [self.W_upmask, self.b_upmask]
    
    def bbprop(self):
        self.lin_bbprop = self.p_y_given_x - self.p_y_given_x*self.p_y_given_x
        self.lin_bbprop /= T.shape(self.p_y_given_x)[0]
        self.dict_bbprop = {}
        self.dict_bbprop.update({self.b_upmask:T.sum(self.lin_bbprop,0)})
        self.dict_bbprop.update({self.W_upmask:T.dot(T.transpose(self.inp*self.inp),self.lin_bbprop)})
        return T.dot(self.lin_bbprop,T.transpose(self.W * self.W)), self.dict_bbprop
    
    def save(self,path):
        f = open(path+'W.pkl','w')
        cPickle.dump(self.W.value,f)
        f.close()
        print >> sys.stderr, self.W, 'saved in %s'%(path+'W.pkl')
        f = open(path+'b.pkl','w')
        cPickle.dump(self.b.value,f)
        f.close()
        print >> sys.stderr, self.b, 'saved in %s'%(path+'b.pkl')
    
    def load(self,path):
        f = open(path+'W.pkl','r')
        self.W.value = cPickle.load(f)
        f.close()
        print >> sys.stderr, self.W, 'loaded from %s'%(path+'W.pkl')
        f = open(path+'b.pkl','r')
        self.b.value = cPickle.load(f)
        f.close()
        print >> sys.stderr, self.b, 'loaded from %s'%(path+'b.pkl')
