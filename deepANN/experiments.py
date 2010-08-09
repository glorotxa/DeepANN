import numpy

import sys
import time
import os
import copy
from pylearn.io import filetensor
import theano
import theano.tensor as T
from jobman import make
import time
from theano.tensor.shared_randomstreams import RandomStreams

def load_mat(fname, save_dir=''):
    print >> sys.stderr, 'loading ndarray from file: ', save_dir + fname
    file_handle = open(os.path.join(save_dir,fname), 'r')
    rval = filetensor.read(file_handle)
    file_handle.close()
    return rval

def definetrainandtest(Model,train,trainaux,valid,validaux,test,testaux,batchsize):
    trai, trai_n = \
    Model.trainfunctionbatch([train,255.,0,theano.config.floatX],None,[trainaux,255.,0,theano.config.floatX],\
                batchsize=batchsize[0])
    val = Model.costfunction([valid,255.,0,theano.config.floatX],None,[validaux,255.,0,theano.config.floatX],\
                batchsize=batchsize[1])
    tes = Model.costfunction([test,255.,0,theano.config.floatX],None,[testaux,255.,0,theano.config.floatX],\
                batchsize=batchsize[1])
    return trai, trai_n, val, tes

def recexpe(state,channel):
    state.timeref=time.time()
    state.seed = 1
    
    state.besttestrec = -1
    state.bestvalrec = -1
    state.besttestvalrec = -1
    
    state.besttestrecct = 0
    state.bestvalrecct = 0
    state.besttestvalrecct = 0
    
    state.resolution = 50000
    state.updatecount = 0

    path = '/mnt/scratch/bengio/glorotxa/data/NIST/'
    
    test = theano.shared(load_mat('Pin07_test_data.ft',path))
    testaux = theano.shared(load_mat('PNIST07_test_data.ft',path))
    #testl= load_mat('Pin07_test_labels.ft',path)
    
    valid = theano.shared(load_mat('Pin07_valid_data.ft',path))
    validaux = theano.shared(load_mat('PNIST07_valid_data.ft',path))
    #validl= load_mat('Pin07_valid_labels.ft',path)
    def trainload(i):
        train = theano.shared(load_mat('Pin07_train%s_data.ft'%(state.seed),path))
        trainaux = theano.shared(load_mat('PNIST07_train%s_data.ft'%state.seed,path))
        #trainl= load_mat('Pin07_train%s_labels.ft'%(state.seed),path)
        return train,trainaux#,trainl
    
    train, trainaux = trainload(state.seed)
    state.Model.rng = numpy.random
    state.Model.theano_rng = RandomStreams()
    state.Model.n_inp = 1024
    state.Model.n_out = 62
    
    Model=make(state.Model)
    
    nepochs = 50
    if state.pretrain == 'pyramide':
        nepochs = nepochs / (1+(Model.depth-1)*2)
    if state.pretrain == 'greedy':
        nepochs = nepochs / (1+Model.depth)
    
    for z in xrange(1,Model.depth+1):
        if z == 1:
            Model.auxiliary(1, -Model.depth+1,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
        if z == 2:
            Model.auxiliary(1, Model.depth-1,1024, auxact = 'sigmoid', auxwdreg = 'l2', \
                    Wtmp = Model.auxlayer.W, btmp = Model.auxlayer.b)
            Model.aux = False
        if state.pretrain != 'no':
            Model.ModeUnsup(z,z-1,noise_lvl = 0.25,update_type='global',lr=state.lr)
            trai, trai_n, val, tes = definetrainandtest(Model,train,trainaux,valid,validaux,test,testaux,(10,100))   
            for i in range(nepochs):
                for j in range(trai_n):
                    if 0:
                        if (state.updatecount % state.resolution) == 0:
                            state.testrec = tes()
                            state.valrec = val()
                            if state.testrec < state.besttestrec or state.besttestrec == -1:
                                state.besttestrec = state.testrec
                                state.besttestrecct = (state.updatecount,state.seed)
                            if state.valrec < state.bestvalrec or state.bestvalrec == -1:
                                state.bestvalrec = state.valrec
                                state.bestvalrecct = (state.updatecount,state.seed)
                                state.besttestvalrec = state.testrec
                                if 'bestmodel' not in os.listdir('.'):
                                    os.mkdir('bestmodel')
                                Model.save('bestmodel/')
                            print >> sys.stderr, '%s: seed %s, test: %s, valid: %s, besttest : %s, bestvalid : %s, besttestval :%s'\
                                    %(state.updatecount,state.seed,state.testrec,state.valrec,\
                                    state.besttestrec,state.bestvalrec,state.besttestvalrec)
                    dum = trai(j)
                    state.updatecount += 1
                state.seed+=1
                train, trainaux = trainload(state.seed)
                channel.save()
        if (state.pretrain in ['greedy','no'] and z == Model.depth) or (state.pretrain == 'pyramide' and z != 1):
            update = state.update
            if z == Model.depth:
                Model.untie()
                nepochs = 99-state.seed+1
                update = 'global'
            Model.aux = True
            Model.ModeAux(z,update_type=update,lr=state.lr)
            trai, trai_n, val, tes = definetrainandtest(Model,train,trainaux,valid,validaux,test,testaux,(10,100))
            
            for i in range(nepochs):
                print >> sys.stderr, 'epoch' , i
                for j in range(trai_n):
                    print >> sys.stderr, 'up' , j
                    if (state.updatecount % state.resolution) == 0:
                        state.testrec = tes()
                        state.valrec = val()
                        if state.testrec < state.besttestrec or state.besttestrec == -1:
                            state.besttestrec = state.testrec
                            state.besttestrecct = (state.updatecount,state.seed)
                        if state.valrec < state.bestvalrec or state.bestvalrec == -1:
                            state.bestvalrec = state.valrec
                            state.bestvalrecct = (state.updatecount,state.seed)
                            state.besttestvalrec = state.testrec
                            if 'bestmodel' not in os.listdir('.'):
                                os.mkdir('bestmodel')
                            Model.save('bestmodel/')
                        print >> sys.stderr, '%s: seed %s, test: %s, valid: %s, besttest : %s, bestvalid : %s, besttestval :%s'\
                                %(state.updatecount,state.seed,state.testrec,state.valrec,\
                                state.besttestrec,state.bestvalrec,state.besttestvalrec)
                    dum = trai(j)
                    state.updatecount += 1
                state.seed+=1
                train, trainaux = trainload(state.seed)
                channel.save()
    return channel.COMPLETE


#-----------------------------------------------------------------------------------------------------------------------
