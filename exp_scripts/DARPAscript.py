try:
    from ANN import *
except ImportError:
    from deepANN.ANN import *
import cPickle
import os
import os.path
import time
import sys

from jobman.tools import DD,expand
from jobman.parse import filemerge

from common.stats import stats

#hardcoded path to your liblinear source:
SVMPATH = '/u/glorotxa/work/NLP/DARPAproject/netscale_sentiment_for_ET/lib/liblinear/' 
SVMRUNALL_PATH = os.path.join(SVMPATH, "run_all")
assert os.access(SVMRUNALL_PATH, os.X_OK)


def rebuildunsup(model,depth,ACT,LR,NOISE_LVL,batchsize,train):
    model.ModeAux(depth+1,update_type='special',noise_lvl=NOISE_LVL,lr=LR)
    if depth > 0:
        givens = {}
        index = T.lscalar()
        givens.update({model.inp : train[index*batchsize:(index+1)*batchsize]})
        if ACT[depth-1] == 'tanh':
            #rescaling between 0 and 1 of the target
            givens.update({model.auxtarget : (model.layers[depth-1].out+1.)/2.})
        else:
            #no rescaling needed if sigmoid
            givens.update({model.auxtarget : model.layers[depth-1].out})
        trainfunc = theano.function([index],model.cost,updates = model.updates, givens = givens)
        testfunc = theano.function([index],model.cost, givens = givens)
        n = train.value.shape[0] / batchsize
        def tes():
            sum=0
            for i in range(train.value.shape[0]/100):
                sum+=testfunc(i)
            return sum/float(i+1)
    else:
        trainfunc,n = model.trainfunctionbatch(train,None,train, batchsize=batchsize)
        tes = model.costfunction(train,None,train, batchsize=100)
    return trainfunc,n,tes

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
    for i in range(10000/500):
        textr = ''
        rep = func(instances[500*i:500*(i+1),:])[0]
        for l in range(rep.shape[0]):
            textr += '%s '%labels[500*i+l]
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    del instances,labels
    f.close()
    print >> sys.stderr, "...done creating libsvm files"
    print >> sys.stderr, stats()

def dosvm(nbinputs,datatrainsave,datatestsave,PATH_SAVE):
    if nbinputs == 100:
        NUMRUNS = 20
    elif nbinputs == 1000:
        NUMRUNS = 10
    elif nbinputs == 10000:
        NUMRUNS = 1
    print >> sys.stderr, 'BEGIN SVM for %s examples'%nbinputs
    C = 0.01
    print >> sys.stderr, C , '-----------------------------------------------------------------'
    print >> sys.stderr, stats()
    os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s'%(SVMRUNALL_PATH,C,nbinputs,NUMRUNS,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
    f = open(PATH_SAVE+'/currentsvm.txt','r')
    a=f.readline()[:-1]
    f.close()
    os.remove(PATH_SAVE+'/currentsvm.txt')
    res = a.split(' ')
    trainerr = [float(res[1])]
    trainerrdev = [float(res[2])]
    testerr = [float(res[3])]
    testerrdev = [float(res[4])]
    Clist=[C]
    
    C = 0.001
    print >> sys.stderr, C  , '-----------------------------------------------------------------'
    print >> sys.stderr, stats()
    os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s'%(SVMRUNALL_PATH,C,nbinputs,NUMRUNS,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
    f = open(PATH_SAVE+'/currentsvm.txt','r')
    a=f.readline()[:-1]
    f.close()
    os.remove(PATH_SAVE+'/currentsvm.txt')
    res = a.split(' ')
    trainerr = [float(res[1])] + trainerr
    trainerrdev = [float(res[2])] + trainerrdev
    testerr = [float(res[3])] + testerr
    testerrdev = [float(res[4])] + testerrdev
    Clist=[C] + Clist
    
    if testerr[1] < testerr[0]:
        C = 0.1        
        while testerr[-1] < testerr[-2] and C<100000:
            print >> sys.stderr, C , '-----------------------------------------------------------------'
            print >> sys.stderr, stats()
            os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s'%(SVMRUNALL_PATH,C,nbinputs,NUMRUNS,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
            f = open(PATH_SAVE+'/currentsvm.txt','r')
            a=f.readline()[:-1]
            f.close()
            os.remove(PATH_SAVE+'/currentsvm.txt')
            res = a.split(' ')
            trainerr += [float(res[1])]
            trainerrdev += [float(res[2])]
            testerr += [float(res[3])]
            testerrdev += [float(res[4])]
            Clist+=[C]
            C=C*10
        if C!=100000:
            return Clist[-2],testerr[-2],testerrdev[-2],trainerr[-2],trainerrdev[-2]
        else:
            return Clist[-1],testerr[-1],testerrdev[-1],trainerr[-1],trainerrdev[-1]
    else:
        C=0.0001
        while testerr[0] < testerr[1] and C>0.000001:
            print >> sys.stderr, C , '-----------------------------------------------------------------'
            print >> sys.stderr, stats()
            os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s'%(SVMRUNALL_PATH,C,nbinputs,NUMRUNS,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
            f = open(PATH_SAVE+'/currentsvm.txt','r')
            a=f.readline()[:-1]
            f.close()
            os.remove(PATH_SAVE+'/currentsvm.txt')
            res = a.split(' ')
            trainerr = [float(res[1])] + trainerr
            trainerrdev = [float(res[2])] + trainerrdev
            testerr = [float(res[3])] + testerr
            testerrdev = [float(res[4])] + testerrdev
            Clist=[C] + Clist
            C=C*0.1
        if C != 0.000001:
            return Clist[1],testerr[1],testerrdev[1],trainerr[1],trainerrdev[1]
        else:
            return Clist[0],testerr[0],testerrdev[0],trainerr[0],trainerrdev[0]

    
def NLPSDAE(state,channel):
    """This script launch a new, or stack on previous, SDAE experiment, training in a greedy layer wise fashion.
    Only tanh and sigmoid activation are supported for stacking, a rectifier activation is possible at the last layer.
    (waiting to validate the best method to stack rectifier on NISTP), it is possible to give a tfidf representation,
    it will then create a softplus auxlayer for the depth 1 unsupervised pre-training with a quadratic reconstruction cost"""
    
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
    VALIDATION_RUNS_FOR_EACH_TRAININGSIZE = state.validation_runs_for_each_trainingsize #list of lists
    VALIDATION_TRAININGSIZE = [i[0] for i in VALIDATION_RUNS_FOR_EACH_TRAININGSIZE] # list
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
    state.besterr = dict([(trainsize, []) for trainsize in VALIDATION_TRAININGSIZE])
    state.besterrepoch = dict([(trainsize, []) for trainsize in VALIDATION_TRAININGSIZE])
    
    if MODEL_RELOAD != None:
        oldstate = expand(DD(filemerge(MODEL_RELOAD+'../current.conf')))
        DEPTH = oldstate.depth + DEPTH
        depthbegin = oldstate.depth
        ACT = oldstate.act + ACT
        N_HID = oldstate.n_hid + N_HID
        NOISE = oldstate.noise + NOISE
        L1 = oldstate.l1 + L1
        L2 = oldstate.l2[:-1] + L2
        NEPOCHS = oldstate.nepochs + NEPOCHS
        LR = oldstate.lr + LR
        NOISE_LVL = oldstate.noise_lvl + NOISE_LVL
        EPOCHSTEST = oldstate.epochstest + EPOCHSTEST
        state.bestrec = oldstate.bestrec
        state.bestrecepoch = oldstate.bestrec
        del oldstate

    if 'rectifier' in ACT:
        assert ACT.index('rectifier')== DEPTH -1 
        # Methods to stack rectifier are still in evaluation (5 different techniques)
        # The best will be implemented in the script soon :).
    f =open(PATH_DATA + NAME_DATATEST + '_1.pkl','r')
    train = theano.shared(numpy.asarray(cPickle.load(f),dtype=theano.config.floatX))
    f.close()
    normalshape = train.value.shape
   
    model=SDAE(numpy.random,RandomStreams(),DEPTH,True,act=ACT,n_hid=N_HID,n_out=5,sparsity=ACTIVATION_REGULARIZATION_COEFF,\
            regularization=WEIGHT_REGULARIZATION_COEFF, wdreg = WEIGHT_REGULARIZATION_TYPE, spreg = ACTIVATION_REGULARIZATION_TYPE, n_inp=NINPUTS,noise=NOISE,tie=True)
    
    #RELOAD previous model
    for i in range(depthbegin):
        print >> sys.stderr, 'reload layer',i+1
        print >> sys.stderr, stats()
        model.layers[i].W.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_W.pkl'%(i+1),'r'))
        model.layers[i].b.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_b.pkl'%(i+1),'r'))
        model.layers[i].mask.value = cPickle.load(open(MODEL_RELOAD + 'Layer%s_mask.pkl'%(i+1),'r'))
    
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
    
    for i in xrange(depthbegin,DEPTH):
        print >> sys.stderr, '-----------------------------BEGIN DEPTH:',i+1
        print >> sys.stderr, stats()
        if i == 0:
            n_aux = NINPUTS
        else:
            n_aux = model.layers[i-1].n_out
        if i==0 and INPUTTYPE == 'tfidf':
            model.depth_max = model.depth_max+1
            model.reconstruction_cost = 'quadratic'
            model.reconstruction_cost_fn = eval('quadratic_cost')
            model.auxiliary(init=1,auxact='softplus',auxdepth=-DEPTH+i+1, auxn_out=n_aux)
        else:
            model.depth_max = model.depth_max+1
            model.reconstruction_cost = 'cross_entropy'
            model.reconstruction_cost_fn = eval('cross_entropy_cost')
            if model.auxlayer != None:
                del model.auxlayer.W
                del model.auxlayer.b
            model.auxiliary(init=1,auxdepth=-DEPTH+i+1, auxn_out=n_aux)
        
        rec = {}
        err = dict([(trainsize, {}) for trainsize in VALIDATION_TRAININGSIZE])
        
        if 0 in EPOCHSTEST[i]: 
            trainfunc,n,tes = rebuildunsup(model,i,ACT,LR[i],None,BATCHSIZE,train)
            createlibsvmfile(model,i,datatrain,datatrainsave)
            createlibsvmfile(model,i,datatest,datatestsave)
            C,testerr,testerrdev,trainerr,trainerrdev = dosvm(100,datatrainsave,datatestsave,PATH_SAVE)
            err100.update({0:(C,testerr,testerrdev,trainerr,trainerrdev)})
            C,testerr,testerrdev,trainerr,trainerrdev = dosvm(1000,datatrainsave,datatestsave,PATH_SAVE)
            err1000.update({0:(C,testerr,testerrdev,trainerr,trainerrdev)})
            C,testerr,testerrdev,trainerr,trainerrdev = dosvm(10000,datatrainsave,datatestsave,PATH_SAVE)
            err10000.update({0:(C,testerr,testerrdev,trainerr,trainerrdev)})
        
        trainfunc,n,tes = rebuildunsup(model,i,ACT,LR[i],NOISE_LVL[i],BATCHSIZE,train)
        if 0 in EPOCHSTEST[i]:
            rec.update({0:tes()})
            
            print >> sys.stderr, '########## INITIAL TEST ############  : '
            print >> sys.stderr, 'CURRENT RECONSTRUCTION ERROR: ',rec[0]
            print >> sys.stderr, 'CURRENT 100 SVM ERROR: ',err100[0]
            print >> sys.stderr, 'CURRENT 1000 SVM ERROR: ',err1000[0]
            print >> sys.stderr, 'CURRENT 10000 SVM ERROR: ',err10000[0]
            print >> sys.stderr, stats()
            
        for cc in range(NEPOCHS[i]):
            time1 = time.time()
            for p in xrange(1,NB_FILES + 1):
                time2=time.time()
                f =open(PATH_DATA + NAME_DATA +'_%s.pkl'%p,'r')
                object = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
                # The last training file is not of the same shape as the other training files.
                # So, to avoid a GPU memory error, we want to make sure it is the same size.
                # In which case, we pad the matrix but keep track of how many n (instances) there actually are.
                # TODO: Also want to pad trainl
                if object.shape == normalshape:
                    train.value = object
                    currentn = normalshape[0]
                    del object
                else:
                    train.value = numpy.concatenate([object,\
                        numpy.zeros((normalshape[0]-object.shape[0],normalshape[1]),dtype=theano.config.floatX)])
                    currentn = object.shape[0]
                    del object
                f.close()
                for j in range(currentn/BATCHSIZE):
                    dum = trainfunc(j)
                print >> sys.stderr, 'File:',p,time.time()-time2, '----'
                print >> sys.stderr, stats()
            print >> sys.stderr, '-----------------------------epoch',cc+1,'time',time.time()-time1
            print >> sys.stderr, stats()
            if cc+1 in EPOCHSTEST[i]:
                trainfunc,n,tes = rebuildunsup(model,i,ACT,LR[i],None,BATCHSIZE,train)
                createlibsvmfile(model,i,datatrain,datatrainsave)
                createlibsvmfile(model,i,datatest,datatestsave)                
                C,testerr,testerrdev,trainerr,trainerrdev = dosvm(100,datatrainsave,datatestsave,PATH_SAVE)
                err100.update({cc+1:(C,testerr,testerrdev,trainerr,trainerrdev)})
                C,testerr,testerrdev,trainerr,trainerrdev = dosvm(1000,datatrainsave,datatestsave,PATH_SAVE)
                err1000.update({cc+1:(C,testerr,testerrdev,trainerr,trainerrdev)})
                C,testerr,testerrdev,trainerr,trainerrdev = dosvm(10000,datatrainsave,datatestsave,PATH_SAVE)
                err10000.update({cc+1:(C,testerr,testerrdev,trainerr,trainerrdev)})
                
                trainfunc,n,tes = rebuildunsup(model,i,ACT,LR[i],NOISE_LVL[i],BATCHSIZE,train)
                f =open(PATH_DATA + NAME_DATATEST +'_1.pkl','r')
                train.value = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
                f.close()
                rec.update({cc+1:tes()})
                print >> sys.stderr, '##########  TEST ############ EPOCH : ',cc+1
                print >> sys.stderr, 'CURRENT RECONSTRUCTION ERROR: ',rec[cc+1]
                print >> sys.stderr, 'CURRENT 100 SVM ERROR: ',err100[cc+1]
                print >> sys.stderr, 'CURRENT 1000 SVM ERROR: ',err1000[cc+1]
                print >> sys.stderr, 'CURRENT 10000 SVM ERROR: ',err10000[cc+1]
                print >> sys.stderr, stats()
                f = open('depth%serr.pkl'%i,'w')
                cPickle.dump(rec,f,-1)
                cPickle.dump(err100,f,-1)
                cPickle.dump(err1000,f,-1)
                cPickle.dump(err10000,f,-1)
                f.close()
                os.mkdir(PATH_SAVE+'/depth%spre%s'%(i+1,cc+1))
                model.save(PATH_SAVE+'/depth%spre%s'%(i+1,cc+1))
        if len(EPOCHSTEST[i])!=0:
            recmin = numpy.min(rec.values())
            for k in rec.keys():
                if rec[k] == recmin:
                    state.bestrec += [recmin]
                    state.bestrecepoch += [k]
            
            errvector = err100.values()
            for k in range(len(errvector)):
                errvector[k] = errvector[k][1]
            errmin = numpy.min(errvector)
            for k in err100.keys():
                if err100[k][1] == errmin:
                    state.besterr100 += [err100[k]]
                    state.besterr100epoch += [k]
            
            errvector = err1000.values()
            for k in range(len(errvector)):
                errvector[k] = errvector[k][1]
            errmin = numpy.min(errvector)
            for k in err1000.keys():
                if err1000[k][1] == errmin:
                    state.besterr1000 += [err1000[k]]
                    state.besterr1000epoch += [k]
            
            errvector = err10000.values()
            for k in range(len(errvector)):
                errvector[k] = errvector[k][1]
            errmin = numpy.min(errvector)
            for k in err10000.keys():
                if err10000[k][1] == errmin:
                    state.besterr10000 += [err10000[k]]
                    state.besterr10000epoch += [k]
        else:
            state.bestrec +=[None]
            state.bestrecepoch += [None]
            state.besterr100 += [None]
            state.besterr100epoch += [None]
            state.besterr1000 += [None]
            state.besterr1000epoch += [None]
            state.besterr10000 += [None]
            state.besterr10000epoch += [None] 
    return channel.COMPLETE
