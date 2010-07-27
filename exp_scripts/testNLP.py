from ANN import *
import cPickle
import os
import time

lr = 0.01
act='tanh'
depth=1
n_hid = 1400
noise = (0.7,0.0041)
l1=0.0
l2=0.0
nepochs = 30
batchsize = 10



f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_1.pkl','r')
train = theano.shared(numpy.asarray(cPickle.load(f),dtype=theano.config.floatX))
print train.value.dtype
f.close()
f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_1.pkl','r')
trainl = theano.shared(cPickle.load(f),name = 'trainl')
f.close()

#dat = pylearn.datasets.MNIST.full()
#train = T.shared(numpy.asarray(dat.train.x,dtype=theano.config.floatX))
#trainl = T.shared(dat.train.y,name = 'trainl')


model=SDAE(numpy.random,RandomStreams(),depth,True,act=act,n_hid=n_hid,n_out=5,sparsity=l1,n_inp=5000,noise='binomial_NLP')
model.auxiliary(1,0,5000)
model.ModeAux(depth,update_type='special',noise_lvl=noise,lr=lr)


g1,n = model.trainfunctionbatch(train,trainl,train, batchsize=batchsize)
tes = model.costfunction(train,trainl,train, batchsize=100)

print tes()

for cc in range(nepochs):
    time1 = time.time()
    for p in xrange(14,16):
        time2 = time.time()
        f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_%s.pkl'%p,'r')
        train.value = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
        f.close()
        f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_%s.pkl'%p,'r')
        trainl.value = cPickle.load(f)
        f.close()
        for i in range(train.value.shape[0]/batchsize):
            dum = g1(i)
        print 'file',p,'time',time.time()-time2
        #if p==14:
        #    print tes() #memmory-bug
    f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_1.pkl','r')
    train.value = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
    f.close()
    f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_1.pkl','r')
    trainl.value = cPickle.load(f)
    f.close()
    print tes()
    print '-----------------------------epoch',cc,'time',time.time()-time1

model.save('/u/glorotxa/depth1ronan/')
