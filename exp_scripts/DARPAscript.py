try:
    from ANN import *
except ImportError:
    from deepANN.ANN import *
import cPickle
import os
import os.path
import time
import sys
import math

from jobman.tools import DD,expand
from jobman.parse import filemerge

from common.stats import stats
from common.str import percent
from common.movingaverage import MovingAverage

# TRAINFUNC is a handle to the model's training function. It is a global
# because it is connected to internal state in the Model. Each time the
# model changes, update TRAINFUNC!
TRAINFUNC       = None
TESTFUNC        = None

globalstate = None

def rebuildunsup(model,depth,ACT,LR,NOISE_LVL,batchsize,train,rule):
    """
    Modify the global TRAINFUNC and TESTFUNC.
    TODO: FIXME! Is it possible not to use global state? If the TRAINFUNC
    and TESTFUNC are connected to model state, then it is unavoidable that
    TRAINFUNC and TESTFUNC should be treated as things with side-effects.
    """

    global TRAINFUNC, TESTFUNC

    model.ModeAux(depth+1,update_type='special',noise_lvl=NOISE_LVL,lr=LR)
    if depth > 0:
        givens = {}
        index = T.lscalar()
        givens.update({model.inp : train[index*batchsize:(index+1)*batchsize]})
        if ACT[depth-1] == 'tanh':
            #rescaling between 0 and 1 of the target
            givens.update({model.auxtarget : (model.layers[depth-1].out+1.)/2.})
        else:
            assert rule != None
            if rule in [1,2,3,5,7,8,9]:
                givens.update({model.auxtarget : T.abs_(model.layers[depth-1].out)})
            if rule == 4:
                givens.update({model.auxtarget : sigmoid_act(model.layers[depth-1].out)})
            if rule in [5,7]:
                cost = (model.auxtarget > 0.0) * (T.log(numpy.sqrt(2.*numpy.pi) * (T.abs_(model.auxsigma)+model.auxsigmamin)) + 0.5 * ((model.auxtarget - model.auxlayer.lin)/ (T.abs_(model.auxsigma)+model.auxsigmamin))**2) - \
                       (model.auxtarget == 0.0) * (T.log(0.5) + T.log(T.erfc((model.auxlayer.lin /  (T.abs_(model.auxsigma)+model.auxsigmamin)) * 1/numpy.sqrt(2.))))
                model.cost = (cost.sum(1)).mean()
                if rule == 5:
                    grad = T.grad(model.cost,model.params + model.auxlayer.params)
	            model.updates = dict((p,p - model.lr * g) for p, g in zip(model.params + model.auxlayer.params , grad))
                    model.updates.update({ model.auxsigma: model.auxsigma - model.lr * 0.01 * T.grad(model.cost,model.auxsigma) })
                if rule == 7:
                    grad = T.grad(model.cost,model.params + model.auxlayer.params)
	            model.updates = dict((p,p - model.lr * g) for p, g in zip(model.params + model.auxlayer.params,grad))
            if rule in [8,9]:
                cost1 = - T.log(0.5) + (model.auxlayer.lin) / (T.abs_(model.auxsigma)+model.auxsigmamin) 
                cost2 = - T.log( 1 - 0.5 * T.exp(  (model.auxlayer.lin) / (T.abs_(model.auxsigma)+model.auxsigmamin) ) )
                cost3 = T.log( 2*(T.abs_(model.auxsigma)+model.auxsigmamin)) + T.abs_(model.auxtarget-model.auxlayer.lin) / (T.abs_(model.auxsigma)+model.auxsigmamin)
                model.cost = ((T.switch(model.auxtarget>0, cost3 , T.switch(model.auxlayer.lin < 0 , cost2 , cost1))).sum(1)).mean()
                if rule == 8:
                    grad = T.grad(model.cost,model.params + model.auxlayer.params)
	            model.updates = dict((p,p - model.lr * g) for p, g in zip(model.params + model.auxlayer.params , grad))
                    model.updates.update({ model.auxsigma: model.auxsigma - model.lr * 0.01 * T.grad(model.cost,model.auxsigma) })
                if rule == 9:
                    grad = T.grad(model.cost,model.params + model.auxlayer.params)
	            model.updates = dict((p,p - model.lr * g) for p, g in zip(model.params + model.auxlayer.params,grad))
            if rule == 6:
                givens.update({model.auxtarget : model.layers[depth-1].out})
        TRAINFUNC = theano.function([index],model.cost,updates = model.updates, givens = givens)
        testfunc = theano.function([index],model.cost, givens = givens)
        n = train.value.shape[0] / batchsize
        def tes():
            sum=0
            for i in range(train.value.shape[0]/globalstate.BATCH_TEST):
                sum+=testfunc(i)
            return sum/float(i+1)
        TESTFUNC = tes
    else:
        TRAINFUNC,n = model.trainfunctionbatch(train,None,train, batchsize=batchsize)
        TESTFUNC = model.costfunction(train,None,train, batchsize=globalstate.BATCH_TEST)

def createlibsvmfile(model,depth,datafiles,dataout):
    print >> sys.stderr, 'Creating libsvm file %s (model=%s, depth=%d, datafiles=%s)...' % (repr(dataout), repr(model),depth,datafiles)
    print >> sys.stderr, stats()
    outputs = [model.layers[depth].out]
    func = theano.function([model.inp],outputs)
    f = open(datafiles[0],'r')
    instances = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
    f.close()
    f = open(datafiles[1],'r')
    labels = numpy.asarray(cPickle.load(f),dtype = 'int64')
    f.close()
    f = open(dataout,'w')
    for i in range(globalstate.NB_MAX_TRAINING_EXAMPLES_SVM/globalstate.BATCH_CREATION_LIBSVM):
        textr = ''
        rep = func(instances[globalstate.BATCH_CREATION_LIBSVM*i:globalstate.BATCH_CREATION_LIBSVM*(i+1),:])[0]
        for l in range(rep.shape[0]):
            textr += '%s '%labels[globalstate.BATCH_CREATION_LIBSVM*i+l]
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    del instances,labels
    f.close()
    print >> sys.stderr, "...done creating libsvm files"
    print >> sys.stderr, stats()

def svm_validation_for_one_trainsize_and_one_C(C, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE):
    """
    Train an SVM using some C on nbinputs training examples, for numrums runs.
    Return:
        testerr,testerrdev,trainerr,trainerrdev
    """
    print >> sys.stderr, "\t\tTraining SVM with C=%f, nbinputs=%d, numruns=%d" % (C, nbinputs,numruns)

    os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s > /dev/null 2> /dev/null'%(globalstate.SVMRUNALL_PATH,C,nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
    results = open(PATH_SAVE+'/currentsvm.txt','r').readline()[:-1].split(' ')
    os.remove(PATH_SAVE+'/currentsvm.txt')
    trainerr       = float(results[1])
    trainerrdev    = float(results[2])
    testerr        = float(results[3])
    testerrdev     = float(results[4])
    trainerrnew    = float(results[5])
    trainerrnewdev = float(results[6])
    testerrnew     = float(results[7])
    testerrnewdev  = float(results[8])
    return testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev


def svm_validation_for_one_trainsize(nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE):
    """
    Train an SVM on nbinputs training examples, for numrums runs.
    Choose the value of C using a linesearch to minimize the testerr.
    Return:
        C,testerr,testerrdev,trainerr,trainerrdev

    MAXSTEPS is the number of steps performed in the line search.
    STEPFACTOR is the initial step size.
    """
    MAXSTEPS=globalstate.SVM_MAXSTEPS
    STEPFACTOR=globalstate.SVM_STEPFACTOR
    INITIALC=globalstate.SVM_INITIALC

    print >> sys.stderr, 'Starting SVM validation for %s examples (numrums=%d, datatrainsave=%s, datatestsave=%s, PATH_SAVE=%s, MAXSTEPS=%d, STEPFACTOR=%f, INITIALC=%f)...' % (nbinputs, numruns,datatrainsave,datatestsave, PATH_SAVE,MAXSTEPS, STEPFACTOR, INITIALC)
    print >> sys.stderr, stats()

    Ccurrent = INITIALC
    Cstepfactor = STEPFACTOR
    Cnew = Ccurrent * Cstepfactor

    C_to_allstats = {}
    Cbest = None

    while len(C_to_allstats) < MAXSTEPS:
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            (testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev) = \
                                                svm_validation_for_one_trainsize_and_one_C(Ccurrent, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE)
            C_to_allstats[Ccurrent] = (testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev)
        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            (testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev) = \
                                                svm_validation_for_one_trainsize_and_one_C(Cnew, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE)
            C_to_allstats[Cnew] = (testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev)
        # If Cnew has a lower test err than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew][0] < C_to_allstats[Ccurrent][0]:
            print >> sys.stderr, "\ttesterr[Cnew %f] = %f < testerr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew][0], Ccurrent, C_to_allstats[Ccurrent][0])
            if Cbest is None or C_to_allstats[Cnew][0] < C_to_allstats[Cbest][0]:
                Cbest = Cnew
                print >> sys.stderr, "\tNEW BEST: Cbest <= %f, testerr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest][0])
            Ccurrent = Cnew
            Cnew *= Cstepfactor
            print >> sys.stderr, "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            print >> sys.stderr, "\ttesterr[Cnew %f] = %f > testerr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew][0], Ccurrent, C_to_allstats[Ccurrent][0])
            if Cbest is None or C_to_allstats[Ccurrent][0] < C_to_allstats[Cbest][0]:
                Cbest = Ccurrent
                print >> sys.stderr, "\tCbest <= %f, testerr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest][0])
            Cstepfactor = 1. / math.sqrt(Cstepfactor)
            Cnew = Ccurrent * Cstepfactor
            print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    for C in allC:
        print >> sys.stderr, "\ttesterr[C %f] = %f" % (C, C_to_allstats[C][0]),
        if C == Cbest: print >> sys.stderr, " *best* (testerr = %f, testerrdev = %f, trainerr = %f, trainerrdev = %f, testerrnew = %f,\
						testerrnewdev = %f, trainerrnew = %f, trainerrnewdev = %f)" % C_to_allstats[C]
        else: print >> sys.stderr, ""
    print >> sys.stderr, '...done with SVM validation for %s examples (numrums=%d, datatrainsave=%s, datatestsave=%s)' % (nbinputs, numruns,datatrainsave,datatestsave)
    print >> sys.stderr, stats()

    return [Cbest] + list(C_to_allstats[Cbest])

def svm_validation(err, reconstruction_error, epoch, model, depth, ACT,LR,NOISE_LVL,BATCHSIZE,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST,RULE):
    """
    Perform full SVM validation.
    """
    global TRAINFUNC

    print >> sys.stderr, "Validating (err=%s,epoch=%s,model=%s,depth=%s,ACT=%s,LR=%s,NOISE_LVL=%s,BATCHSIZE=%s,train=%s,datatrain=%s,datatrainsave=%s,datatest=%s,datatestsave=%s,VALIDATION_TRAININGSIZE=%s,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE=%s,PATH_SAVE=%s)..." % (err, epoch, model, depth, ACT,LR,NOISE_LVL,BATCHSIZE,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE)
    print >> sys.stderr, stats()

    # Call with noiselevel = None before running the SVM.
    # No noise because we want the exact representation for each instance.
    rebuildunsup(model,depth,ACT,LR,None,BATCHSIZE,train,RULE)

    createlibsvmfile(model,depth,datatrain,datatrainsave)
    createlibsvmfile(model,depth,datatest,datatestsave)

    for trainsize in VALIDATION_TRAININGSIZE:
        print trainsize
        print VALIDATION_RUNS_FOR_EACH_TRAININGSIZE
        C,testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev =\
                                            svm_validation_for_one_trainsize(trainsize,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE[`trainsize`],datatrainsave,datatestsave,PATH_SAVE)
        err[trainsize].update({epoch:(C,testerr,testerrdev,trainerr,trainerrdev,testerrnew,testerrnewdev,trainerrnew,trainerrnewdev)})


    if epoch != 0:
        f = open(PATH_DATA + NAME_DATATEST +'_1.pkl','r')
        train.container.value[:] = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
        f.close()

    # Now, restore TRAINFUNC with the original NOISE_LVL
    rebuildunsup(model,depth,ACT,LR,NOISE_LVL,BATCHSIZE,train,RULE)
    reconstruction_error.update({epoch:TESTFUNC()})

    print >> sys.stderr, 'VALIDATION: depth %d / epoch %d / reconstruction error (is this on test or train?): ' % (depth+1, epoch),reconstruction_error[epoch]
    for trainsize in VALIDATION_TRAININGSIZE:
        print >> sys.stderr, 'VALIDATION: depth %d / epoch %d / trainsize %d / svm error' % (depth+1, epoch, trainsize),err[trainsize][epoch]
    print >> sys.stderr, stats()

    if epoch != 0:
        f = open('depth%serr.pkl'%depth,'w')
        cPickle.dump(reconstruction_error,f,-1)
        for trainsize in VALIDATION_TRAININGSIZE:
            cPickle.dump(err[trainsize],f,-1)
        f.close()
        modeldir = os.path.join(PATH_SAVE, 'depth%spre%s' % (depth+1,epoch))
        if not os.path.isdir(modeldir):
            os.mkdir(modeldir)
        model.save(modeldir)
        if RULE == 5:
            f = open(modeldir + '/auxsigma.pkl','w')
            cPickle.dump(model.auxsigma.value,f,-1)
            f.close()

    print >> sys.stderr, "...done validating (err=%s,epoch=%s,model=%s,depth=%s,ACT=%s,LR=%s,NOISE_LVL=%s,BATCHSIZE=%s,train=%s,datatrain=%s,datatrainsave=%s,datatest=%s,datatestsave=%s,VALIDATION_TRAININGSIZE=%s,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE=%s,PATH_SAVE=%s)" % (err, epoch, model, depth, ACT,LR,NOISE_LVL,BATCHSIZE,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE)
    print >> sys.stderr, stats()


def NLPSDAE(state,channel):
    """This script launch a new, or stack on previous, SDAE experiment, training in a greedy layer wise fashion.
    Only tanh and sigmoid activation are supported for stacking, a rectifier activation is possible at the last layer.
    (waiting to validate the best method to stack rectifier on NISTP), it is possible to give a tfidf representation,
    it will then create a softplus auxlayer for the depth 1 unsupervised pre-training with a quadratic reconstruction cost"""

    global globalstate
    globalstate = state
    globalstate.SVMRUNALL_PATH = os.path.join(globalstate.SVMPATH, "run_all")

    print >> sys.stderr, globalstate.SVMRUNALL_PATH
    assert os.access(globalstate.SVMRUNALL_PATH, os.X_OK)

    # Hyper-parameters
    LR = state.lr#list
    ACT = state.act #list
    DEPTH = state.depth
    N_HID = state.n_hid #list
    NOISE = state.noise #list
    NOISE_LVL = state.noise_lvl#list
    ACTIVATION_REGULARIZATION_TYPE = state.activation_regularization_type
    ACTIVATION_REGULARIZATION_COEFF = state.activation_regularization_coeff #list
    WEIGHT_REGULARIZATION_TYPE = state.weight_regularization_type
    WEIGHT_REGULARIZATION_COEFF = state.weight_regularization_coeff #list
    NEPOCHS = state.nepochs #list
    VALIDATION_RUNS_FOR_EACH_TRAININGSIZE = state.validation_runs_for_each_trainingsize #dict from trainsize string to number of validation runs at this training size
    VALIDATION_TRAININGSIZE = [int(trainsize) for trainsize in VALIDATION_RUNS_FOR_EACH_TRAININGSIZE] # list
    VALIDATION_TRAININGSIZE.sort()
    EPOCHSTEST = state.epochstest #list
    BATCHSIZE = state.batchsize
    PATH_SAVE = channel.remote_path if hasattr(channel,'remote_path') else channel.path
    NB_FILES = state.nb_files
    PATH_DATA = state.path_data
    NAME_DATA = state.name_traindata
    NAME_LABEL = state.name_trainlabel
    NAME_DATATEST = state.name_testdata
    NAME_LABELTEST = state.name_testlabel
    MODEL_RELOAD = state.model_reload if hasattr(state,'model_reload') else None
    NINPUTS = state.ninputs          # Number of input dimensions
    INPUTTYPE = state.inputtype
    RULE = state.rule if hasattr(state,'rule') else None
    RandomStreams(state.seed)
    numpy.random.seed(state.seed)
    datatrain = (PATH_DATA+NAME_DATA+'_1.pkl',PATH_DATA+NAME_LABEL+'_1.pkl')
    datatrainsave = PATH_SAVE+'/train.libsvm'
    datatest = (PATH_DATA+NAME_DATATEST+'_1.pkl',PATH_DATA+NAME_LABELTEST+'_1.pkl')
    datatestsave = PATH_SAVE+'/test.libsvm'

    depthbegin = 0

    #monitor best performance for reconstruction and classification
    state.bestrec = []
    state.bestrecepoch = []
    state.besterr = dict([(`trainsize`, []) for trainsize in VALIDATION_TRAININGSIZE])
    state.besterrepoch = dict([(`trainsize`, []) for trainsize in VALIDATION_TRAININGSIZE])

    if MODEL_RELOAD != None:
        oldstate = expand(DD(filemerge(MODEL_RELOAD+'../current.conf')))
        DEPTH = oldstate.depth + DEPTH
        depthbegin = oldstate.depth
        ACT = oldstate.act + ACT
        N_HID = oldstate.n_hid + N_HID
        NOISE = oldstate.noise + NOISE
        ACTIVATION_REGULARIZATION_COEFF = oldstate.activation_regularization_coeff + ACTIVATION_REGULARIZATION_COEFF
        WEIGHT_REGULARIZATION_COEFF = oldstate.weight_regularization_coeff[:-1] + WEIGHT_REGULARIZATION_COEFF
        NEPOCHS = oldstate.nepochs + NEPOCHS
        LR = oldstate.lr + LR
        NOISE_LVL = oldstate.noise_lvl + NOISE_LVL
        EPOCHSTEST = oldstate.epochstest + EPOCHSTEST
        state.bestrec = oldstate.bestrec
        state.bestrecepoch = oldstate.bestrec
        del oldstate

    #if 'rectifier' in ACT:
        #assert ACT.index('rectifier')== DEPTH -1
        # Methods to stack rectifier are still in evaluation (5 different techniques)
        # The best will be implemented in the script soon :).
    filename = PATH_DATA + NAME_DATATEST + '_1.pkl'
    print filename
    f =open(filename,'r')
    train = theano.shared(numpy.asarray(cPickle.load(f),dtype=theano.config.floatX))
    f.close()
    normalshape = train.value.shape
    
    model=SDAE(numpy.random,RandomStreams(),DEPTH,True,act=ACT,n_hid=N_HID,n_out=5,sparsity=ACTIVATION_REGULARIZATION_COEFF,\
            regularization=WEIGHT_REGULARIZATION_COEFF, wdreg = WEIGHT_REGULARIZATION_TYPE, spreg = ACTIVATION_REGULARIZATION_TYPE, n_inp=NINPUTS,noise=NOISE,tie=True)

    #RELOAD previous model
    for depth in range(depthbegin):
        print >> sys.stderr, 'reload layer',depth+1
        print >> sys.stderr, stats()
        model.layers[depth].W.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_W.pkl'%(depth+1),'r'))
        model.layers[depth].b.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_b.pkl'%(depth+1),'r'))
        model.layers[depth].mask.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_mask.pkl'%(depth+1),'r'))

    state.act = ACT
    state.depth = DEPTH
    state.depthbegin = depthbegin
    state.n_hid = N_HID
    state.noise = NOISE
    state.activation_regularization_coeff = ACTIVATION_REGULARIZATION_COEFF
    state.weight_regularization_coeff = WEIGHT_REGULARIZATION_COEFF
    state.nepochs = NEPOCHS
    state.LR = LR
    state.noise_lvl = NOISE_LVL
    state.epochstest = EPOCHSTEST
    channel.save()


    for depth in xrange(depthbegin,DEPTH):
        print >> sys.stderr, 'BEGIN DEPTH %s...' % (percent(depth+1, DEPTH))
        print >> sys.stderr, stats()
        if depth == 0:
            n_aux = NINPUTS
        else:
            n_aux = model.layers[depth-1].n_out
        if depth==0 and INPUTTYPE == 'tfidf':
            model.depth_max = model.depth_max+1
            model.reconstruction_cost = 'quadratic'
            model.reconstruction_cost_fn = quadratic_cost
            model.auxiliary(init=1,auxact='softplus',auxdepth=-DEPTH+depth+1, auxn_out=n_aux)
        else:
            model.depth_max = model.depth_max+1
            if depth == 0 or ACT[depth-1] != 'rectifier':
                model.reconstruction_cost = 'cross_entropy'
                model.reconstruction_cost_fn = cross_entropy_cost
                if model.auxlayer != None:
                    del model.auxlayer.W
                    del model.auxlayer.b
                model.auxiliary(init=1,auxdepth=-DEPTH+depth+1, auxn_out=n_aux)
            else:
		assert RULE != None
                if RULE in [1,2,4]:
                    model.reconstruction_cost = 'cross_entropy'
                    model.reconstruction_cost_fn = cross_entropy_cost
                    if model.auxlayer != None:
                        del model.auxlayer.W
                        del model.auxlayer.b
                    model.auxiliary(init=1,auxdepth=-DEPTH+depth+1, auxn_out=n_aux)
                    rebuildunsup(model,depth,ACT,LR[depth],None,BATCHSIZE,train,RULE)
                    if RULE in [1,2]:
                        rescal_rectifier_model(model,depth,PATH_DATA,NAME_DATA,NB_FILES,RULE)
                if RULE in [3,6]:
                    model.reconstruction_cost = 'quadratic'
                    model.reconstruction_cost_fn = quadratic_cost
                    if model.auxlayer != None:
                        del model.auxlayer.W
                        del model.auxlayer.b
                    if RULE == 3:
                        model.auxiliary(init=1,auxdepth=-DEPTH+depth+1, auxact='softplus',auxn_out=n_aux)
                    if RULE == 6:
                        model.auxiliary(init=1,auxdepth=-DEPTH+depth+1, auxact='lin',auxn_out=n_aux)
                if RULE in [5,7,8,9]:
                    model.auxiliary(init=1,auxdepth=-DEPTH+depth+1, auxn_out=n_aux)
                    stdvect = compute_representation_std(model,depth,PATH_DATA,NAME_DATA,NB_FILES)
                    model.auxsigma = theano.shared(value = numpy.asarray(stdvect, dtype = theano.config.floatX), name = 'auxsigma')
                    model.auxsigmamin = theano.shared(value = numpy.asarray( 0.001 * numpy.ones((n_aux,)), dtype = theano.config.floatX), name = 'auxsigmamin')

        reconstruction_error = {}
        err = dict([(trainsize, {}) for trainsize in VALIDATION_TRAININGSIZE])

        rebuildunsup(model,depth,ACT,LR[depth],NOISE_LVL[depth],BATCHSIZE,train,RULE)

        state.currentdepth = depth

        epoch = 0
        if epoch in EPOCHSTEST[depth]:
            svm_validation(err, reconstruction_error, epoch, model, depth,ACT,LR[depth],NOISE_LVL[depth],BATCHSIZE,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST,RULE)
            channel.save()

        train_reconstruction_error_mvgavg = MovingAverage()
        for epoch in xrange(1,NEPOCHS[depth]+1):
            time1 = time.time()
            state.currentepoch = epoch
            for filenb in xrange(1,NB_FILES + 1):
                print >> sys.stderr, "\t\tAbout to read file %s..." % percent(filenb, NB_FILES)
                print >> sys.stderr, "\t\t", stats()
#                initial_file_time = time.time()
                f =open(PATH_DATA + NAME_DATA +'_%s.pkl'%filenb,'r')
                object = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
                print >> sys.stderr, "\t\t...read file %s" % percent(filenb, NB_FILES)
                print >> sys.stderr, "\t\t", stats()
                # The last training file is not of the same shape as the other training files.
                # So, to avoid a GPU memory error, we want to make sure it is the same size.
                # In which case, we pad the matrix but keep track of how many n (instances) there actually are.
                # TODO: Also want to pad trainl
                if object.shape == normalshape:
                    train.container.value[:] = object
                    currentn = normalshape[0]
                    del object
                else:
                    train.container.value[:] = numpy.concatenate([object,\
                        numpy.zeros((normalshape[0]-object.shape[0],normalshape[1]),dtype=theano.config.floatX)])
                    currentn = object.shape[0]
                    del object
                f.close()
                if train.value.min() < 0:
                    print >> sys.stderr, "WARNING: Negative input, currently input should be positive"
                if train.value.max() > 1. and INPUTTYPE!='tfidf':
                    print >> sys.stderr, "WARNING: Some inputs are > 1, without tfidf inputtype, it should be in the range [0,1]" 
                for j in range(currentn/BATCHSIZE):
                    reconstruction_error_over_batch = TRAINFUNC(j)
                    train_reconstruction_error_mvgavg.add(reconstruction_error_over_batch)
#                    print reconstruction_error_over_batch
#                current_file_time = time.time()
#                print >> sys.stderr, 'File:',filenb,time.time()-time2, '----'
                print >> sys.stderr, "\t\tAt depth %d, epoch %d, finished training over file %s, training pre-update reconstruction error %s" % (depth, epoch, percent(filenb, NB_FILES), train_reconstruction_error_mvgavg)
                print >> sys.stderr, "\t\t", stats()
            print >> sys.stderr, '...finished training epoch #%s' % percent(epoch, NEPOCHS[depth])
            print >> sys.stderr, stats()
#           sys.stderr.flush()
#           or maybe you need
            #jobman cachesync
            
            if epoch in EPOCHSTEST[depth]:
                svm_validation(err, reconstruction_error, epoch, model,depth,ACT,LR[depth],NOISE_LVL[depth],BATCHSIZE,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST,RULE)

            channel.save()
        if len(EPOCHSTEST[depth])!=0:
            recmin = numpy.min(reconstruction_error.values())
            for k in reconstruction_error.keys():
                if reconstruction_error[k] == recmin:
                    state.bestrec += [recmin]
                    state.bestrecepoch += [k]

            for trainsize in VALIDATION_TRAININGSIZE:
                errvector = err[trainsize].values()
                for k in range(len(errvector)):
                    errvector[k] = errvector[k][1]
                errmin = numpy.min(errvector)
                for k in err[trainsize].keys():
                    if err[trainsize][k][1] == errmin:
                        state.besterr[`trainsize`] += [err[trainsize][k]]
                        state.besterrepoch[`trainsize`] += [k]
        else:
            state.bestrec +=[None]
            state.bestrecepoch += [None]
            for trainsize in VALIDATION_TRAININGSIZE:
                state.besterr[`trainsize`] += [None]
                state.besterrepoch[`trainsize`] += [None]
        print >> sys.stderr, '...DONE DEPTH %s' % (percent(depth+1, DEPTH - depthbegin))
        print >> sys.stderr, stats()
    return channel.COMPLETE

def rescal_rectifier_model(model,depth,PATH_DATA,NAME_DATA,NB_FILES,rule):
    print >> sys.stderr, "Rescaling of the rectifier model following the rule: %s"%rule
    print >> sys.stderr, stats()
    outputs = [model.layers[depth-1].out]
    func = theano.function([model.inp],outputs)
    max_value = numpy.zeros((1,model.n_hid[depth-1]))
    for filenb in xrange(1,NB_FILES + 1):
        f =open(PATH_DATA + NAME_DATA +'_%s.pkl'%filenb,'r')
        instances = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
        f.close()
        for i in range(instances.shape[0]/globalstate.BATCH_CREATION_LIBSVM):
            rep = numpy.abs(func(instances[globalstate.BATCH_CREATION_LIBSVM*i:globalstate.BATCH_CREATION_LIBSVM*(i+1),:])[0])
            max_value = numpy.asarray([numpy.concatenate([max_value,rep]).max(0)])
    del instances
    if rule == 2:
        model.layers[depth-1].W.container.value[:] =  \
		numpy.asarray((model.layers[depth-1].W.value.T / max_value).T,dtype=theano.config.floatX)
        model.layers[depth-1].b.container.value[:] =  \
		numpy.asarray((model.layers[depth-1].b.value / max_value[0,:]),dtype=theano.config.floatX)
    if rule == 1:
        model.layers[depth-1].W.container.value[:] =  model.layers[depth-1].W.value / max_value.max()
        model.layers[depth-1].b.container.value[:] =  model.layers[depth-1].b.value / max_value.max()
    print >> sys.stderr, "...done rescaling parameters"
    print >> sys.stderr, stats()


def compute_representation_std(model,depth,PATH_DATA,NAME_DATA,NB_FILES):
    print >> sys.stderr, "Computing representation std for sigma initialization"
    print >> sys.stderr, stats()
    outputs = [model.layers[depth-1].out]
    func = theano.function([model.inp],outputs)
    sumvector = numpy.zeros((1,model.n_hid[depth-1]))
    count = 0
    for filenb in xrange(1,NB_FILES + 1):
        f =open(PATH_DATA + NAME_DATA +'_%s.pkl'%filenb,'r')
        instances = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
        f.close()
        for i in range(instances.shape[0]/globalstate.BATCH_CREATION_LIBSVM):
            count += globalstate.BATCH_CREATION_LIBSVM
            rep = numpy.abs(func(instances[globalstate.BATCH_CREATION_LIBSVM*i:globalstate.BATCH_CREATION_LIBSVM*(i+1),:])[0])
            sumvector += rep.sum(0)
    meanvector = sumvector / float(count)
    sumvector = numpy.zeros((1,model.n_hid[depth-1]))
    count = 0
    for filenb in xrange(1,NB_FILES + 1):
        f =open(PATH_DATA + NAME_DATA +'_%s.pkl'%filenb,'r')
        instances = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
        f.close()
        for i in range(instances.shape[0]/globalstate.BATCH_CREATION_LIBSVM):
            count += globalstate.BATCH_CREATION_LIBSVM
            rep = (numpy.abs(func(instances[globalstate.BATCH_CREATION_LIBSVM*i:globalstate.BATCH_CREATION_LIBSVM*(i+1),:])[0]) - meanvector)**2
            sumvector += rep.sum(0)
    stdvector = numpy.sqrt(sumvector / float(count))
    del instances
    print >> sys.stderr, "...done computing std"
    print >> sys.stderr, stats()
    return stdvector.reshape((model.n_hid[depth-1],))


