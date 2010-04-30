import numpy

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
    print 'loading ndarray from file: ', save_dir + fname
    file_handle = open(os.path.join(save_dir,fname), 'r')
    rval = filetensor.read(file_handle)
    file_handle.close()
    return rval

def definetrainandtest(train,trainaux,valid,validaux,test,testaux):
    trai, trai_n = \
    model.trainfunctionbatch([train,255.,0,theano.config.floatX],None,[trainaux,255.,0,theano.config.floatX],\
                batchsize=batchsize)
    val = Model.costfunction([valid,255.,0,theano.config.floatX],None,[validaux,255.,0,theano.config.floatX],\
                batchsize=100)
    tes = Model.costfunction([test,255.,0,theano.config.floatX],None,[testaux,255.,0,theano.config.floatX],\
                batchsize=100)
    return trai, trai_n, val, tes

def P07_rec(state,channel):
    state.timeref=time.time()
    state.seed = 1
    
    state.besttestrec = -1
    state.bestvalidrec = -1
    state.besttestvalidrec = -1
    
    state.besttestrecct = 0
    state.bestvalidrecct = 0
    state.besttestvalidrecct = 0
    
    state.resolution = 40000
    state.updatecount = 0

    path = '/mnt/scratch/bengio/glorotxa/data/NIST/'
    
    test = load_mat('Pin07_test_data.ft',path)
    testaux = load_mat('PNIST07_test_data.ft',path)
    #testl= load_mat('Pin07_test_labels.ft',path)
    
    valid = load_mat('Pin07_valid_data.ft',path)
    validaux = load_mat('PNIST07_valid_data.ft',path)
    #validl= load_mat('Pin07_valid_labels.ft',path)
    def trainload(i):
        train = load_mat('Pin07_train%s_data.ft'%(state.seed),path)
        trainaux = load_mat('PNIST07_train%s_data.ft'%state.seed,path)
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
    mini = 1
    if state.pretrain == 'no':
        mini = Model.depth
    for z in xrange(1,Model.depth+1):
        if z == 1:
            Model.auxiliary(1, -Model.depth+1,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
        if z == 2:
            Model.auxiliary(1, Model.depth-1,1024, auxact = 'sigmoid', auxwdreg = 'l2', \
                    Wtmp = Model.auxlayer.W, btmp = Model.auxlayer.b)
            Model.aux = False
        if state.pretrain != 'no':
            
            Model.ModeUnsup(z,z-1,update_type='global',lr=0.001)
            trai, trai_n, val, tes = definetrainandtest(train,trainaux,valid,validaux,test,testaux)
            
            for i in range(nepochs):
                for j in range(trai_n):
                    if z==1:
                        if (state.updatecout % state.resolution) == 0:
                            state.testrec = tes()
                            state.valrec = val()
                            if tmptest < state.besttestrec or state.besttestrec == -1:
                                state.besttestrec = tmptest
                                state.besttestrecct = (state.updatecount,state.seed)
                            if tmpval < state.bestvalrec or state.bestvalrec == -1:
                                state.bestvalrec = tmpval
                                state.bestvalrecct = (state.updatecount,state.seed)
                                state.besttestvalrec = tmptest
                                if 'bestmodel' not in os.listdir('.'):
                                    os.mkdir('bestmodel')
                                Model.save('bestmodel/')
                    dum = trai[j]
                    state.updatecount += 1
                state.seed+=1
                train, trainaux = trainload(state.seed)
                channel.save()
        
        if z != 1 or (state.pretrain == 'greedy' and z == Model.depth):
            update = state.update
            if z == Model.depth:
                nepochs = 99-state.seed+1
                update = 'global'
            Model.aux = True
            Model.ModeAux(z,update_type=update,lr=0.001)
            trai, trai_n, val, tes = definetrainandtest(train,trainaux,valid,validaux,test,testaux)
            
            for i in range(nepochs):
                for j in range(trai_n):
                    if (state.updatecout % state.resolution) == 0:
                        state.testrec = tes()
                        state.valrec = val()
                        if tmptest < state.besttestrec or state.besttestrec == -1:
                            state.besttestrec = tmptest
                            state.besttestrecct = (state.updatecount,state.seed)
                        if tmpval < state.bestvalrec or state.bestvalrec == -1:
                            state.bestvalrec = tmpval
                            state.bestvalrecct = (state.updatecount,state.seed)
                            state.besttestvalrec = tmptest
                            if 'bestmodel' not in os.listdir('.'):
                                os.mkdir('bestmodel')
                            Model.save('bestmodel/')
                    dum = trai[j]
                    state.updatecount += 1
                state.seed+=1
                train, trainaux = trainload(state.seed)
                channel.save()
    return channel.COMPLETE


#-----------------------------------------------------------------------------------------------------------------------