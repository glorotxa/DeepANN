#definition of the erf et cerf tensor OP.

from theano.scalar.basic import UnaryScalarOp,exp,sqrt,upgrade_to_float,complex_types,float_types
import theano.tensor.elemwise as elemwise
import scipy.special
import numpy

class Erf(UnaryScalarOp):
    def impl(self, x):
        return scipy.special.erf(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            return gz * 2. / numpy.sqrt(numpy.pi) * exp(-x*x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erf(%(x)s);" % locals()

erf = elemwise.Elemwise(Erf(upgrade_to_float, name = 'Terf'))


class Erfc(UnaryScalarOp):
    def impl(self, x):
        return scipy.special.erfc(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            return - gz * 2. / numpy.sqrt(numpy.pi) * exp(-x*x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

erfc = elemwise.Elemwise(Erfc(upgrade_to_float, name = 'Terfc'))


if __name__ == '__main__':
    import theano
    import theano.tensor as T
    a = T.matrix()
    atmp = numpy.random.uniform(size=(10,10))
    b = erf(a)
    c = erfc(a)
    d = T.grad(b.sum(),a)
    e = T.grad(c.sum(),a)
    f=theano.function([a],[b,c,d,e])
    btmp,ctmp,dtmp,etmp=f(atmp)
    print abs(btmp - scipy.special.erf(atmp)).sum()
    print abs(ctmp - scipy.special.erfc(atmp)).sum()
    print abs(dtmp - (2/numpy.sqrt(numpy.pi) * numpy.exp(-atmp*atmp))).sum()
    print abs(etmp + (2/numpy.sqrt(numpy.pi) * numpy.exp(-atmp*atmp))).sum()
