try:
    from ANN import *
except ImportError:
    from deepANN.ANN import *
import cPickle
import os
import time

lr = 0.0001
act=['tanh']
depth=1
n_hid = [1400]
noise = (0.7,0.0041)
l1=0.0
l2=0.0
nepochs = 30
batchsize = 10
path2savemodel = '/u/glorotxa/'



f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_1.pkl','r')
train = theano.shared(numpy.asarray(cPickle.load(f),dtype=theano.config.floatX))
print train.value.dtype
f.close()
f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_1.pkl','r')
trainl = theano.shared(cPickle.load(f),name = 'trainl')
f.close()


model=SDAE(numpy.random,RandomStreams(),depth,True,act=act,n_hid=n_hid,n_out=5,sparsity=l1,n_inp=5000,noise='binomial',tie=False)
model.auxiliary(1,0,2500)
model.ModeAux(depth,update_type='special',noise_lvl=noise,lr=lr)

#model.layers[0].W.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer1_W.pkl','r'))
#model.layers[0].b.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer1_b.pkl','r'))
#model.layers[0].mask.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer1_mask.pkl','r'))
#model.layers[1].W.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer2_W.pkl','r'))
#model.layers[1].b.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer2_b.pkl','r'))
#model.layers[1].mask.value = cPickle.load(open('/u/glorotxa/depth2ronan/Layer2_mask.pkl','r'))

#givens = {}
#index = T.lscalar()
#givens.update({model.inp : train[index*batchsize:(index+1)*batchsize]})
#givens.update({model.auxtarget : (model.layers[1].out+1.)/2.})
#g1 = theano.function([index],[model.cost],updates = model.updates, givens = givens)

#def tes():
#    sum=0
#    for i in range(train.value.shape[0]/100):
#        sum+=g1(i)[0]
#    return sum/float(i+1)

g1,n = model.trainfunctionbatch(train,trainl,train , batchsize=batchsize)
tes = model.costfunction(train,trainl,train, batchsize=100)

normalshape = train.value.shape

print tes()

for cc in range(nepochs):
    time1 = time.time()
    for p in xrange(1,16):
        time2 = time.time()
        f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_%s.pkl'%p,'r')
        object = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
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
        f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_%s.pkl'%p,'r')
        trainl.value = cPickle.load(f)
        f.close()
        for i in range(currentn/batchsize):
            dum = g1(i)
        print 'file',p,'time',time.time()-time2
    f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_train_1.pkl','r')
    train.value = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
    f.close()
    f =open('/u/glorotxa/work/NLP/DARPAproject/NEC/OpenTable5000_trainl_1.pkl','r')
    trainl.value = cPickle.load(f)
    f.close()
    print tes()
    print '-----------------------------epoch',cc,'time',time.time()-time1

#model.save(path2save)
