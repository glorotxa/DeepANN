#
# This support 4 dataset for now: MNIST, CIFAR10, ImageNet and shapesetbatch.
# for shapeset you need pygame and http://github.com/glorotxa/Shapeset




import numpy

import time
import os
import copy
from pylearn.io import filetensor
import theano
import theano.tensor as T
from jobman import make
import time

#specifics to MNIST

import pylearn.datasets.MNIST

from deepANN.ANN import *

def createcurridata(state,seed,batchsize):
    if batchsize == None:
        batchsize = state.batchsize
    return Curridatacreate(state.curridata.nfeatures,Polygongen,state.curridata.genparams,\
                    [buildimage,buildedges,builddepthmap,buildidentity,buildsegmentation,output,buildedgesanglec],\
                    state.curridata.dependencies,state.curridata.funcparams,batchsize,seed)

def convertout(out):
    target =    0*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==0)) +\
                1*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==0)) +\
                2*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==1)) +\
                3*((out[:,0]==1) * (out[:,1] == 1) * (out[:,2]==0)) +\
                4*((out[:,0]==0) * (out[:,1] == 1) * (out[:,2]==1)) +\
                5*((out[:,0]==1) * (out[:,1] == 0) * (out[:,2]==1)) +\
                6*((out[:,0]==2) * (out[:,1] == 0) * (out[:,2]==0)) +\
                7*((out[:,0]==0) * (out[:,1] == 2) * (out[:,2]==0)) +\
                8*((out[:,0]==0) * (out[:,1] == 0) * (out[:,2]==2))
    return target

def save_mat(fname, mat, save_dir=''):
    assert isinstance(mat, numpy.ndarray)
    print 'save ndarray to file: ', save_dir + fname
    file_handle = open(os.path.join(save_dir, fname), 'w')
    filetensor.write(file_handle, mat)
    writebool = False
    while not writebool:
        try:
            file_handle.close()
            writebool = True
        except:
            print 'save model error'
            time.sleep((numpy.random.randint(10)+2)*10)

def load_mat(fname, save_dir=''):
    print 'loading ndarray from file: ', save_dir + fname
    file_handle = open(os.path.join(save_dir,fname), 'r')
    rval = filetensor.read(file_handle)
    file_handle.close()
    return rval

def reorderdata(length,seed=None):
    if seed is not None:
        numpy.random.seed(seed)
    neworder = numpy.random.permutation(length)
    return neworder

def loaddata(state):
    if state.dat == 'MNIST':
        state.in_size = 784
        state.n_out = 10
        dat = pylearn.datasets.MNIST.full()
        test = theano.shared(numpy.asarray(dat.test.x,dtype=theano.config.floatX),name='test')
        testl= theano.shared(numpy.asarray(copy.deepcopy(dat.test.y),dtype='int32'),name='testl')
        valid = theano.shared(numpy.asarray(dat.valid.x,dtype=theano.config.floatX),name='valid')
        validl = theano.shared(numpy.asarray(copy.deepcopy(dat.valid.y),dtype='int32'),name='validl')
        train = theano.shared(numpy.asarray(dat.train.x,dtype=theano.config.floatX),name='train')
        trainl = theano.shared(numpy.asarray(copy.deepcopy(dat.train.y),dtype='int32'),name='trainl')
        del dat
    elif state.dat == 'ImageNet':
        state.in_size = 37*37
        state.n_out = 10
        train=theano.shared(numpy.asarray(load_mat('Imagenettrain.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='train')
        trainl=theano.shared(numpy.asarray(load_mat('Imagenettrainl.ft','/home/glorotxa/data/'),dtype='int32'),name='trainl')
        test=theano.shared(numpy.asarray(load_mat('Imagenettest.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='test')
        testl=theano.shared(numpy.asarray(load_mat('Imagenettestl.ft','/home/glorotxa/data/'),dtype='int32'),name='testl')
        valid=theano.shared(numpy.asarray(load_mat('Imagenetvalid.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='valid')
        validl=theano.shared(numpy.asarray(load_mat('Imagenetvalidl.ft','/home/glorotxa/data/'),dtype='int32'),name='validl')
    elif state.dat == 'CIFAR10':
        state.in_size = 32*32*3
        state.n_out = 10
        train=theano.shared(numpy.asarray(load_mat('Cifar10train.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='train')
        trainl=theano.shared(numpy.asarray(load_mat('Cifar10trainl.ft','/home/glorotxa/data/'),dtype='int32'),name='trainl')
        test=theano.shared(numpy.asarray(load_mat('Cifar10test.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='test')
        testl=theano.shared(numpy.asarray(load_mat('Cifar10testl.ft','/home/glorotxa/data/'),dtype='int32'),name='testl')
        valid=theano.shared(numpy.asarray(load_mat('Cifar10valid.ft','/home/glorotxa/data/')/255.0,dtype=theano.config.floatX),name='valid')
        validl=theano.shared(numpy.asarray(load_mat('Cifar10validl.ft','/home/glorotxa/data/'),dtype='int32'),name='validl')
    elif state.dat == 'shapesetbatch':
        from buildfeaturespolygon import buildimage,buildedges,builddepthmap,buildidentity,buildsegmentation,output,buildedgesanglec
        from polygongen import Polygongen
        from curridata import Curridata as Curridatacreate

        state.in_size = 32*32
        state.n_out = 9
        curridatatrain=createcurridata(state,1,100000)
        train=theano.shared(numpy.asarray(curridatatrain.image,name='train',dtype=theano.config.floatX))
        trainl=theano.shared(numpy.asarray(convertout(curridatatrain.output),dtype='int32'),name='trainl')
        del curridatatrain
        curridatatest=createcurridata(state,0,10000)
        test=theano.shared(numpy.asarray(curridatatest.image,name='test',dtype=theano.config.floatX))
        testl=theano.shared(numpy.asarray(convertout(curridatatest.output),dtype='int32'),name='testl')
        del curridatatest
        curridatavalid=createcurridata(state,-1,10000)
        valid=theano.shared(numpy.asarray(curridatavalid.image,name='valid',dtype=theano.config.floatX))
        validl=theano.shared(numpy.asarray(convertout(curridatavalid.output),dtype='int32'),name='validl')
        del curridatavalid
    return test,testl,valid,validl,train,trainl

def pretraining(state,channel):
    import theano,pylearn.version,deepANN
    pylearn.version.record_versions(state,[theano,pylearn,deepANN])
    
    theano.config.floatX = state.floatX
    theano.config.device = state.device
    theano.sandbox.cuda.use(theano.config.device)

    print "floatX",theano.config.floatX
    print "device",theano.config.device

    test,testl,valid,validl,train,trainl = loaddata(state)
    
    numpy.random.seed(state.seed)
    
    model=SDAE(numpy.random,RandomStreams(),state.depth,True,act=state.act,n_out=state.n_out,n_hid=state.n_hid,n_inp=state.in_size,maskbool = None)
    
    #greedy layer-wise pre-training
    for i in range(state.depth):
        model.ModeUnsup(i+1,i,update_type='global',noise_lvl=state.noise,lr=state.unsup_lr)
        trainfunc,n = model.trainfunctionbatch(train,trainl,None,state.batchsize)
        testfunc = model.costfunction(test,testl,batchsize=50)
        state.rec=testfunc()
        print state.rec
        for k in range(state.nbepochs_unsup):
            print 'depth',i,'epoch',k 
            for j in range(n):
                dum = trainfunc(j)
            state.rec=testfunc()
            print state.rec
            neworder = reorderdata(train.value.shape[0])
            train.value = train.value[neworder,:]
            trainl.value= trainl.value[neworder]
    model.save('.')
    
    channel.save()
    return channel.COMPLETE
   
def finetuning(state,channel):
    import theano,pylearn.version,deepANN
    pylearn.version.record_versions(state,[theano,pylearn,deepANN])
    
    theano.config.floatX = state.floatX
    theano.config.device = state.device
    theano.sandbox.cuda.use(theano.config.device)
    
    print "floatX",theano.config.floatX
    print "device",theano.config.device
    
    test,testl,valid,validl,train,trainl = loaddata(state)
    numpy.random.seed(state.seed)
    
    model=SDAE(numpy.random,RandomStreams(),state.depth,True,act=state.act,n_out=state.n_out,n_hid=state.n_hid,n_inp=state.in_size,maskbool = None)
    if hasattr(state,'loadpath'):
        model.load(state.loadpath)
    model.ModeSup(state.depth,state.depth,update_type='global',lr=state.sup_lr)
    trainfunc,n = model.trainfunctionbatch(train,trainl,None,state.batchsize)
    theano.printing.debugprint(trainfunc)
    testfunc = model.errorfunction(test,testl,batchsize=50)
    validfunc = model.errorfunction(valid,validl,batchsize=50)
    
    state.testerr = testfunc()
    state.validerr = validfunc()
    state.besttesterr = state.testerr
    state.besttesterrct = 0
    state.besttestvaliderr = state.testerr
    state.bestvaliderr = state.validerr
    state.bestvalidct = 0
    print 'finetuning'
    #finetuning
    test_err = []
    valid_err = []
    for i in range(state.nbepochs_sup):
        for j in range(n):
            dum = trainfunc(j)
        state.testerr = testfunc()
        state.validerr = validfunc()
        test_err.append(state.testerr)
        valid_err.append(state.validerr)
        if state.testerr < state.besttesterr:
            state.besttesterr = state.testerr
            state.besttesterrct = i+1
        if state.validerr < state.bestvaliderr:
            state.bestvaliderr = state.validerr
            state.bestvaliderrct = i+1
            state.besttestvaliderr = state.testerr
            model.save('.')
        print 'epoch', i , 'test', state.testerr, 'valid' ,state.validerr
        neworder = reorderdata(train.value.shape[0])
        train.value = train.value[neworder,:]
        trainl.value = trainl.value[neworder]
        channel.save()
        test_f = open("err.pkl",'w')
        cPickle.dump(("test_error","valid_error"),test_f,-1)
        cPickle.dump(test_err,test_f,-1)
        cPickle.dump(valid_err,test_f,-1)
        test_f.close()
    return channel.COMPLETE
