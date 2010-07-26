#
# You can also submit the jobs on the command line like this example:
# jobman sqlschedules <tablepath> Experimentsbatchpretrain.finetuning dat="MNIST" depth=1 tie=True "n_hid={{100,250,500,750,1000}}" act=tanh "sup_lr={{.01,.001,.0001,.00001}}" seed=1 nbepochs_sup=1000 batchsize=10
#


from jobman.tools import DD, flatten
from jobman import sql
from jobman.parse import filemerge
from Experimentsbatchpretrain import *
import numpy

db = sql.db('postgres://glorotxa@gershwin.iro.umontreal.ca/glorotxa_db/pretrainexpe')

state = DD()
state.curridata = DD(filemerge('Curridata.conf'))

state.depth = 3
state.tie = True
state.n_hid = 1000 #nb of unit per layer
state.act = 'tanh'

state.sup_lr = 0.01
state.unsup_lr = 0.001
state.noise = 0.25

state.seed = 1

state.nbepochs_unsup = 30 #maximal number of supervised updates
state.nbepochs_sup = 1000 #maximal number of unsupervised updates per layer
state.batchsize = 10

for i in ['MNIST','CIFAR10','ImageNet','shapesetbatch']:
    state.dat =i 
    sql.insert_job(pretrain, flatten(state), db)

db.createView('pretrainexpeview')
