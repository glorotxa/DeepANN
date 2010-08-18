#!/usr/bin/env python
"""
Import unlabelled data (pkl files containing a numpy array), and perform
a random projection on it.
We will create a subdirectory and save the projected matrix to a
corresponding .pkl file in that subdir.

We might also scale the values and then squash them.

USAGE:
    ./randomprojection.py [-d dimensions] [-s seed] file1.pkl ...

    e.g. ./randomprojection.py -d 1000 /u/glorotxa/work/NLP/DARPAproject/*instances_*pkl

NOTE:
    * Right now, we only use a Gaussian random matrix.
    * With the deterministic "online" random version, the runtime
    typically scales in the number of different features.
    * With "batch" MODE, we cache the entire random matrix in
    memory. WARNING: However, this might not be deterministic. So try
    to do all your random projections in one invocation.

REQUIREMENTS:
    * http://github.com/turian/pyrandomprojection
        [optional, only for "online" MODE]
    * http://github.com/turian/common
    * NumPy
"""

from common.stats import stats
from common.str import percent

import sys
import cPickle
import numpy
import os
import os.path

MODE = "online"     # Deterministic, low-memory version
#MODE = "batch"     # Cache the entire random matrix in memory. (However,
                    # this might not be deterministic. So try to do all
                    # your random projections in one invocation.)
                    # Could be, in principle, slower than online if the
                    # number of entries >> #nonzeros.

#SQUASH="sigmoid"   # "none", "erf", or "sigmoid"
SQUASH="erf"   # "none", "erf", or "sigmoid"
#SCALE_BEFORE_SQUASH=1/5.7
SCALE_BEFORE_SQUASH=None

RANDOMIZATION_TYPE="gaussian"
                    # No other option is currently supported

if MODE == "online":
    import pyrandomprojection
elif MODE == "batch":
    # Cache the random matrix in memory
    randommatrix = None


#def project(dict, dimensions, RANDOMIZATION_TYPE="gaussian", TERNARY_NON_ZERO_PERCENT=None, RANDOM_SEED=0):

def batchproject(x, dimensions, seed, randomization_type):
    # Batch (cached, high-memory) random projection
    global randommatrix

    if randommatrix is None:
        print >> sys.stderr, "Creating random matrix of shape %s" % `(x.shape[1], dimensions)`
        print >> sys.stderr, stats()
        numpy.random.seed(seed)
        assert randomization_type == "gaussian"
        randommatrix = numpy.random.normal(size=(x.shape[1], dimensions))
    else:
        assert randommatrix.shape == (x.shape[1], dimensions)       # We assume the projection matrix won't change

    print >> sys.stderr, "Multiplying x by random matrix..."
    print >> sys.stderr, stats()
    newx = numpy.dot(x, randommatrix)
    print >> sys.stderr, "...done multiplying x by random matrix"
    print >> sys.stderr, stats()
    
    return newx


def onlineproject(x, dimensions, seed, randomization_type):
    # Online (low-memory) random projection
    
    newx = numpy.zeros((x.shape[0], dimensions))

    nonzeros = x.nonzero()      # (list of rows, list of cols) of all nonzeros
    
    # (col, row) of all nonzeros
    # We reorder like this so that we can group all columns together, and look up the randomrow for each column feature only once.
    nonzero_colrow = [(nonzeros[1][l], nonzeros[0][l]) for l in range(len(nonzeros[0]))]
    nonzero_colrow.sort()
    nonzero_colrow.reverse()
    
    randrow_key = None
    randrow_values = None
    randrows_computed = 0
    for l, (col, row) in enumerate(nonzero_colrow):
        if randrow_key != col:
            randrow_key = col
            randrow_values = pyrandomprojection.randomrow(key=col, dimensions=dimensions, RANDOMIZATION_TYPE=randomization_type, RANDOM_SEED=seed)

            randrows_computed += 1
            if randrows_computed % 500 == 0:
                print >> sys.stderr, "Retrieved %s random rows thus far, done with %s of nonzeroes on %s..." % (percent(randrows_computed, x.shape[1]), percent(l+1, len(nonzero_colrow)), f)
                print >> sys.stderr, stats()
        newrow = x[row,col] * randrow_values
        assert newx[row].shape == newrow.shape
        newx[row] += newrow
#        if (l+1) % 10000 == 0:
#            print >> sys.stderr, "Done with %s of nonzeroes on %s..." % (percent(l+1, len(nonzero_colrow)), f)
#            print >> sys.stderr, stats()
    return newx

def project(x, dimensions, seed, randomization_type, mode=MODE):
    if mode == "batch": return batchproject(x, dimensions, seed, randomization_type)
    elif mode == "online": return onlineproject(x, dimensions, seed, randomization_type)
    else: assert 0

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dimensions", dest="dimensions", default=1000, type="int", help="number of dimensions in random output")
    parser.add_option("-s", "--seed", dest="seed", default=0, type="int", help="random seed")
    (options, args) = parser.parse_args()

    import random
    random.seed(options.seed)
    numpy.random.seed(options.seed)

    assert len(args) > 0    # You need to pass in pkl files to project.

    for i, f in enumerate(args):
        print >> sys.stderr, "\nLoading %s (file %s)..." % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()
        x = cPickle.load(open(f, "rb"))
        print >> sys.stderr, "...loading %s (file %s)" % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()

        assert x.ndim == 2
        print >> sys.stderr, "Read instance matrix with shape %s, creating projection with shape %s" % (x.shape, (x.shape[0], options.dimensions))

        newx = project(x, dimensions=options.dimensions, seed=options.seed, randomization_type=RANDOMIZATION_TYPE, mode=MODE)
        assert newx.shape == (x.shape[0], options.dimensions)

        if SCALE_BEFORE_SQUASH == None:
            SCALE_BEFORE_SQUASH = 1. / newx.std()
            print  >> sys.stderr, "Setting SCALE_BEFORE_SQUASH to %f on the basis of %s" % (SCALE_BEFORE_SQUASH, f)

        newx *= SCALE_BEFORE_SQUASH

        print >> sys.stderr, "After scaling by %f but before squashing using %s, newx mean = %f and stddev = %f" % (SCALE_BEFORE_SQUASH, SQUASH, numpy.mean(newx), numpy.std(newx))

        if SQUASH == "none":
            pass
        elif SQUASH == "erf":
            import scipy.special
            newx = (scipy.special.erf(newx) + 1.)/2.
        elif SQUASH == "sigmoid":
            newx = 1.0 / (1.0 + numpy.exp(-newx))
        else:
            assert 0

        print >> sys.stderr, "After scaling by %f and squashing using %s, newx mean = %f and stddev = %f" % (SCALE_BEFORE_SQUASH, SQUASH, numpy.mean(newx), numpy.std(newx))

        newdir = os.path.join(os.path.dirname(f), "randomprojection.dimensions=%d.seed=%d.randomization=%s.mode=%s.scale=%f.squash=%s" % (options.dimensions, options.seed, RANDOMIZATION_TYPE, MODE, SCALE_BEFORE_SQUASH, SQUASH))
        newf = os.path.join(newdir, os.path.basename(f))

        if not os.path.isdir(newdir):
            print >> sys.stderr, "Creating directory %s..." % newdir
            os.mkdir(newdir)
        assert os.path.isdir(newdir)

        print >> sys.stderr, "Writing new pickle file to %s..." % newf
        print >> sys.stderr, stats()
        cPickle.dump(newx, open(newf, "wb"))
        print >> sys.stderr, "...done writing new pickle file to %s" % newf
        print >> sys.stderr, stats()
