""" This script plot and save the different curve of interest for the DARPAscript results.
it must be launched in the folder containing the current.conf file and will create the corresponding
plots: SVM error with respect to number of epochs for all the different number of training exemples
for all layer in the same graph and reconstruction error. """
  
import cPickle
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy

from jobman.tools import DD,expand
from jobman.parse import filemerge

rec = {}
err = {}
epochvect = {}

state = expand(DD(filemerge('orig.conf')))
val_run_per_tr_siz = state.validation_runs_for_each_trainingsize
val_run = [int(trainsize) for trainsize in val_run_per_tr_siz ]
val_run.sort()
epochstest = state.epochstest 
nepochs = state.nepochs

depth = state.depth

for i in range(depth):
    err.update({i:{}})
    rec.update({i:[]})
    epochvect.update({i:[]})
    for j in val_run:
        err[i].update({j:[]})


epochoffset = 0

for i in range(depth):
    f = open('depth%serr.pkl'%i,'r')
    rectmp = cPickle.load(f)
    errtmp = {}
    for j in val_run:
        errtmp.update({j:cPickle.load(f)})
    for j in numpy.sort(epochstest[i]):
	if j in rectmp.keys():
            epochvect[i] += [j+epochoffset]
            rec[i] += [rectmp[j]]
            for k in val_run:
                err[i][k] += [errtmp[k][j][1]]
    epochoffset+=nepochs[i]

for j in range(depth):
    pylab.figure(1)    
    pylab.plot(epochvect[j],rec[j])
    ct = 2
    for i in val_run:
        pylab.figure(ct)
        pylab.plot(epochvect[j],err[j][i])
        ct+=1

a=[]
for i in range(depth):
    a+= ['Layer %s'%(i+1)]

pylab.figure(1)
pylab.xlabel('Unsupervised epochs')    
pylab.ylabel('Reconstruction error')
pylab.legend(a)
pylab.savefig('Reconstruction.png')
ct = 2
for i in val_run:
    pylab.figure(ct)
    pylab.xlabel('Unsupervised epochs')
    pylab.ylabel('SVM error for %s examples'%i)
    pylab.legend(a)
    pylab.savefig('SVM%s.png'%i)
    ct+=1
#pylab.show()

