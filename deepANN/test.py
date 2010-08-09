
from ANN import *

theano.config.mode = 'FAST_RUN'

import pylearn.datasets.MNIST
import sys
import copy, os
from pylearn.io import filetensor


def load_mat(fname, save_dir=''):
    print >> sys.stderr, 'loading ndarray from file: ', save_dir + fname
    file_handle = open(os.path.join(save_dir,fname), 'r')
    rval = filetensor.read(file_handle)
    file_handle.close()
    return rval

path1 = '/data/lisa/data/ift6266h10/data/'

testaux = theano.shared(load_mat('PNIST07_test_data.ft', path1))
validaux = theano.shared(load_mat('PNIST07_valid_data.ft', path1))
valid = theano.shared(load_mat('Pin07_valid_data.ft', path1))
test = theano.shared(load_mat('Pin07_test_data.ft', path1))
testl = theano.shared(load_mat('PNIST07_test_labels.ft', path1))
validl = theano.shared(load_mat('PNIST07_valid_labels.ft', path1))


trainaux = theano.shared(load_mat('PNIST07_train1_data.ft', path1))
trainl = theano.shared(load_mat('PNIST07_train1_labels.ft', path1))
path2 = '/data/lisa8/data/ift6266h10/data/'
train = theano.shared(load_mat('Pin07_train1_data.ft', path2))

batchsize = 10

a1=SDAE(numpy.random,RandomStreams(),act='tanh',depth=2,one_sided_in=True,n_hid=1000,
            regularization=False,sparsity=False,reconstruction_cost='cross_entropy',
            n_inp=1024,n_out=62,tie=True,bbbool = False,maskbool = None)

#a1.auxiliary(1, -3,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
#a1.auxiliary(1, -2,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
#a1.auxiliary(1, -1,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
#a1.auxiliary(1, 0,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
#a1.auxiliary(1,1,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
#a1.auxiliary(1, 2,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)
a1.auxiliary(1, 3,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = None, btmp = None)



a1.ModeAux(4,update_type='global',lr=0.001)
g1,n = a1.trainfunctionbatch([train,255.,0,theano.config.floatX],trainl,[trainaux,255.,0,theano.config.floatX],\
                batchsize=batchsize)

val = a1.costfunction([valid,255.,0,theano.config.floatX],trainl,[validaux,255.,0,theano.config.floatX],\
                batchsize=100)

tes = a1.costfunction([test,255.,0,theano.config.floatX],trainl,[testaux,255.,0,theano.config.floatX],\
                batchsize=100)


val = a1.errorfunction([valid,255.,0,theano.config.floatX],validl,batchsize=100)

tes = a1.errorfunction([test,255.,0,theano.config.floatX],testl,batchsize=100)


for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i

a1.auxiliary(1, 2,1024, auxact = 'sigmoid', auxwdreg = 'l2', Wtmp = a1.auxlayer.W, btmp = a1.auxlayer.b)
a1.aux = False
a1.ModeUnsup(2,1,update_type='global',lr=0.001)
g1,n = a1.trainfunctionbatch([train,255.,theano.config.floatX],None,[trainl,255.,theano.config.floatX],\
                batchsize=batchsize)

for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i

a1.aux = True
a1.ModeAux(2,update_type='special',lr=0.001)
g1,n = a1.trainfunctionbatch([train,255.,theano.config.floatX],trainlabels,[trainl,255.,theano.config.floatX],\
                batchsize=batchsize)

for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i


a1.ModeUnsup(3,2,update_type='global',lr=0.001)
g1,n = a1.trainfunctionbatch([train,255.,theano.config.floatX],None,[trainl,255.,theano.config.floatX],\
                batchsize=batchsize)

for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i


a1.ModeAux(3,update_type='special',lr=0.001)
g1,n = a1.trainfunctionbatch([train,255.,theano.config.floatX],None,[trainl,255.,theano.config.floatX],\
                batchsize=batchsize)

for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i

for i in range(n):
    print >> sys.stderr, g1(i),i

import pygame
import numpy
pygame.surfarray.use_arraytype('numpy')

screen = pygame.Surface((4*32,4*32*3),depth=32)
it = 0
for i in range(1000):
    screen.fill(0)
    x = test.value[i,:]
    im = numpy.asarray([numpy.reshape(x,(32,32))]*3).T
    new=pygame.surfarray.make_surface(im)
    new2 = pygame.transform.scale(new, (32*4, 32*4))
    screen.blit(new2,(0,0))
    x = testaux.value[i,:]
    im = numpy.asarray([numpy.reshape(x,(32,32))]*3).T
    new=pygame.surfarray.make_surface(im)
    new2 = pygame.transform.scale(new, (32*4, 32*4))
    screen.blit(new2,(0,4*32*2))
    x = a1.auxrepresentation(test.value[i:i+1,:]/255.)[0,:]
    im = numpy.asarray([numpy.reshape(x,(32,32))*255]*3).T
    new=pygame.surfarray.make_surface(im)
    new2 = pygame.transform.scale(new, (32*4, 32*4))
    screen.blit(new2,(0,4*32*1))
    pygame.image.save(screen,'/u/glorotxa/bestmodel/%s.PNG'%i)
    raw_input('')



#dat = pylearn.datasets.MNIST.full()
#test = theano.shared(value = numpy.asarray((dat.test.x),dtype = theano.config.floatX),name='test')
#testl = T.cast(theano.shared(value = numpy.asarray(dat.test.y,dtype = theano.config.floatX),name='testl'),'int32')
#valid = theano.shared(value = numpy.asarray((dat.valid.x),dtype = theano.config.floatX),name='valid')
#validl = T.cast(theano.shared(value = numpy.asarray(dat.valid.y,dtype = theano.config.floatX),name='validl'),'int32')
#train = theano.shared(value = numpy.asarray((dat.train.x),dtype = theano.config.floatX),name='train')
#trainlstor=theano.shared(value = numpy.asarray(dat.train.y,dtype = theano.config.floatX),name='trainl')
#trainl = T.cast(trainlstor,'int32')
#del dat


#def training(a1,a2,train,trainl,test,testl,maxi = 2,b=False,batchsize=10):
    #g1,n = a1.trainfunctionbatch(train,trainl,batchsize=batchsize)
    #g2,n = a2.trainfunctionbatch(train,trainl,batchsize=batchsize)
    ##theano.printing.pydotprint(g, outfile='/u/glorotxa/youp2.png')
    #err = 1
    #if b:
        #print >> sys.stderr, 1
        #f1 = a1.errorfunction(test,testl)
        #print >> sys.stderr, 2
        #f2 = a2.errorfunction(test,testl)
    #else:
        #print >> sys.stderr, 1
        #f1 = a1.costfunction(test,testl)
        #print >> sys.stderr, 2
        #f2 = a2.costfunction(test,testl,test)
    #err1 = [f1()]
    #err2 = [f2()]
    #print >> sys.stderr, a2.layers[-1].W_upmask.value.min(),a2.layers[-1].W_upmask.value.max(),\
            #a2.layers[-2].W_upmask.value.min(),a2.layers[-2].W_upmask.value.max()
    #epoch = 0
    #errref1 = 10000000000
    #errref2 = 10000000000
    #t=time.time()
    #while epoch<=maxi:
        #for i in range(n):
            #print >> sys.stderr, g1(i),g2(i), 'epoch:', epoch, 'err:',errref1,errref2 , i
        #epoch += 1
        #err1 += [f1()]
        #err2 += [f2()]
        #if err1[-1]<errref1:
            #errref1 = err1[-1]
        #if err2[-1]<errref2:
            #errref2 = err2[-1]
    #tc=time.time()-t
    #return tc
    #order = numpy.random.permutation(train.value.shape[0])
    #train.value = train.value[order,:]
    #trainlstor.value = trainlstor.value[order]

#import time

#a.ModeUnsup(1,0,0.25,'global',lr=0.0005)
#t1=training(a,train,trainl,test,testl,maxi = 10,batchsize=10)
#print >> sys.stderr, t1

#a.ModeUnsup(2,1,0.25,'global',lr=0.0005)
#t2=training(a,train,trainl,test,testl,maxi = 10,batchsize=10)
#print >> sys.stderr, t2

#a.ModeUnsup(1,0,0.25,'global',lr=0.0005,unsup_scaling=0.5)
#t3=training(a,train,trainl,test,testl,maxi = 10,batchsize=10)
#print >> sys.stderr, t3
