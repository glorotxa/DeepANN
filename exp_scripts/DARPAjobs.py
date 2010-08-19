from jobman.tools import DD, flatten
from jobman import sql
from DARPAscript import NLPSDAE

db = sql.db('postgres://glorotxa@gershwin.iro.umontreal.ca/glorotxa_db/opentablegpu') # you should change this line to match the database you need


state = DD()

state.act = ['tanh']
state.depth = 1
state.n_hid = [5000]

state.noise = ['binomial_NLP']

state.weight_regularization_type = 'l2'
state.weight_regularization_coeff = [0.0,0.0]
state.activation_regularization_type = 'l1'

# Number of pretraining epochs, one-per-layer
state.nepochs = [30]

# Different validation runs
#        - 100 training examples (x20 different samples of 100 training examples)
#        - 1000 training examples (x10 different samples of 1000 training examples)
#        - 10000 training examples (x1 different sample of 10000 training examples)
# (because of jobman, the keys have to be strings, not ints)
# NOTE: Probably you don't want to make trainsize larger than 10K,
# because it will be too large for CPU memory.
state.validation_runs_for_each_trainingsize = {"100": 20, "1000": 10, "10000": 1}

# For each layer, a list of the epochs at which you evaluate the
# reconstruction error and linear-SVM-supervised error.
# All the different results you have from here will be stored in a
# separate file per layer.
state.epochstest = [[0,1,2,3,4,6,8,11,15,30]]
#epochstest = [[0,5,30],[0,5,30],[0,2,4,8,16,30]]

state.BATCH_TEST = 100
state.BATCH_CREATION_LIBSVM = 500
state.NB_MAX_TRAINING_EXAMPLES_SVM = 10000
#NB_MAX_TRAINING_EXAMPLES_SVM = 1000     # FIXME: Change back to 10000 <========================================================================
                                        # 1000 is just for fast running during development
#NB_MAX_TRAINING_EXAMPLES_SVM = 100     # FIXME: Change back to 10000 <========================================================================
#                                        # 100 is just for superfast running during development

state.SVM_INITIALC    = 0.001
state.SVM_STEPFACTOR  = 10.
state.SVM_MAXSTEPS    = 10

#hardcoded path to your liblinear source:
state.SVMPATH = '/work/glorotxa/netscale_sentiment_for_ET/lib/liblinear/'

state.batchsize = 10

# The total number of files into which the training set is broken
state.nb_files = 15
state.path_data = '/scratch/glorotxa/OpenTable/'
# Train and test (validation) here should be disjoint subsets of the
# original full training set.
state.name_traindata = 'OpenTable_5000_train_instances'
state.name_trainlabel =  'OpenTable_5000_train_labels'
state.name_testdata = 'OpenTable_5000_test_instances'
state.name_testlabel = 'OpenTable_5000_test_labels'

# If there is a model file specified to build upon, the output of this
# model is the input for the model we are currently building.
state.model_to_build_upon = None

state.ninputs = 5000

# inputtype ('binary', 'tfidf', other options?) determines what the
# decoding activation function is for the first layer
# e.g. inputtype 'tfidf' ('tf*idf'?) uses activation function softplus
# to decode the tf*idf.
state.inputtype = 'binary'

state.seed = 123

#here is the for loops that does the grid:

for i in [0.01,0.001,0.0001]:
    state.lr = [i]
    for j in [(0.7,0.0041),(0.5,0.003),(0.8,0.005)]:
        state.noise_lvl=[j]
        for k in [0.001,0.00001,0.0]:
            state.activation_regularization_coeff = [k]
	    sql.insert_job(NLPSDAE, flatten(state), db) #this submit the current state DD to the db, if it already exist in the db no additionnal job is added.


db.createView('opentablegpuview')

#in order to access the db from a compute node you need to create an tunnel ssh connection on ang23:
#(to do one time, I think you should keep the shell open or you can create the tunnel on a screen and detached it)

#ssh -v -f -o ServerAliveInterval=60 -o ServerAliveCountMax=60 -N -L *:5432:localhost:5432 gershwin.iro.umontreal.ca

#you will need to give your LISA password. 

#here is the command you use to launch 1 jobs of the db.
#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32  sqsub -q gpu -r 2d -n 1 --gpp=1 --memperproc=2.5G -o the_output_you_want jobman sql 'postgres://glorotxa@ang23/glorotxa_db/opentablegpu' /scratch/glorotxa/
