"""

This script define the different symbolic reconstruction cost function

"""
import theano.tensor as T
import theano

# costs utils:---------------------------------------------------
# in order to fix numerical instability of the cost and gradient calculation for the cross entropy we calculate it
# with the following functions direclty from the activation:

def sigmoid_cross_entropy(target, output_act, ddXE):
    XE = target * (- T.log(1 + T.exp(-output_act))) + (1 - target) * (- T.log(1 + T.exp(output_act)))
    return [[-T.mean(T.sum(XE, axis=1),axis=0)], ddXE]

def tanh_cross_entropy(target, output_act, ddXE):
    XE = target * (- T.log(1 + T.exp(- 2*output_act))) + (1 - target) * (- T.log(1 + T.exp( 2*output_act)))
    return [[-T.mean(T.sum(XE, axis=1),axis=0)], ddXE]

def abstanh_cross_entropy(target, output_act, ddXE):
    XE = target * (- T.log(1 + T.exp(- 2*T.abs_(output_act)))) + (1 - target) * (- T.log(1 + T.exp( 2*T.abs_(output_act))))
    return [[-T.mean(T.sum(XE, axis=1),axis=0)], ddXE]

def tanhnorm_cross_entropy(target, output_act, ddXE):
    XE = target * (- T.log(1 + T.exp(- 2*0.6666*output_act))) + (1 - target) * (- T.log(1 + T.exp( 2*0.6666*output_act)))
    return [[-T.mean(T.sum(XE, axis=1),axis=0)], ddXE]

def abstanhnorm_cross_entropy(target, output_act, ddXE):
    XE = target * (- T.log(1 + T.exp(- 2*0.6666*T.abs_(output_act)))) + \
                    (1 - target) * (- T.log(1 + T.exp(2*0.6666*T.abs_(output_act))))
    return [[-T.mean(T.sum(XE, axis=1),axis=0)], ddXE]


def cross_entropy_cost(target, output, output_act, in_sided, out_sided, in_bounded, out_bounded, act):
    assert in_bounded
    #assert out_bounded
    scale_bb = 1.
    if in_bounded != 1.:
        target = target / in_bounded
    #if out_bounded != 1.:
    #    output = output / out_bounded
    #    scale_bb = 1. / out_bounded
    if not in_sided:
        target = (target+1)/(2.0)
    if not out_sided:
        output= (output+1)/(2.0)
        scale_bb = scale_bb / 2.
    ddXE = target * scale_bb * 1./(output * output) + (1 - target) * scale_bb * 1./((1 - output) * (1-output))
    ddXE /= T.shape(ddXE)[0]
    ddXE = T.cast(ddXE,dtype=theano.config.floatX)
    if act in ['sigmoid','tanh','tanhnorm','abstanh','abstanhnorm']:
        if act == 'sigmoid':
            return sigmoid_cross_entropy(target, output_act,ddXE)
        if act == 'tanh':
            return tanh_cross_entropy(target, output_act,ddXE)
        if act == 'tanhnorm':
            return tanhnorm_cross_entropy(target, output_act,ddXE)
        if act == 'abstanh':
            return abstanh_cross_entropy(target, output_act,ddXE)
        if act == 'abstanhnorm':
            return abstanhnorm_cross_entropy(target, output_act,ddXE)
    else:
        XE = target * T.log(output) + (1 - target) * T.log(1 - output)
        return [[-T.mean(T.sum(XE, axis=1),axis=0)] , ddXE]

def quadratic_cost(target, output, output_act, in_sided, out_sided, in_bounded, out_bounded, act ):
    if in_sided == True:
        assert out_sided == True
    #@TO THINK: previous case not critical, but does it really make sense?
    if out_sided == True:
        assert in_sided == True
    if out_bounded != False:
        assert in_bounded <= out_bounded
    ddSQ = 2 * T.ones_like(output)
    ddSQ /= T.shape(ddSQ)[0]
    return [[T.mean(T.sum(T.sqr(output - target), axis=1),axis=0)], ddSQ]
