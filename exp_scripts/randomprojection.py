#!/usr/bin/env python
"""
Import unlabelled data (pkl files containing a numpy array), and perform a random projection on it.
Save to a corresponding .pkl file.

USAGE:
    ./randomprojection.py [-d dimensions] [-s seed] file1.pkl ...

    e.g. ./randomprojection.py -d 1000 /u/glorotxa/work/NLP/DARPAproject/*instances*pkl

NOTE:
    * Right now, we only use a Gaussian random matrix.
    * We cache the entire random matrix in memory. (However, this might
    not be deterministic. So try to do all your random projections in
    one invocation.) The deterministic online random version is very slow.

REQUIREMENTS:
    * http://github.com/turian/pyrandomprojection
        [optional, only if you don't want to hold the random matrix in memory]
    * http://github.com/turian/common
    * NumPy
"""

#import pyrandomprojection
from common.stats import stats
from common.str import percent

import sys
import cPickle
import numpy

#def project(dict, dimensions, RANDOMIZATION_TYPE="gaussian", TERNARY_NON_ZERO_PERCENT=None, RANDOM_SEED=0):

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

    # Cache the random matrix in memory
    randommatrix = None

    for i, f in enumerate(args):
        print >> sys.stderr, "\nLoading %s (file %s)..." % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()
        x = cPickle.load(open(f, "rb"))
        print >> sys.stderr, "...loading %s (file %s)" % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()

        assert x.ndim == 2
        newx = numpy.zeros((x.shape[0], options.dimensions))
        print >> sys.stderr, "Read instance matrix with shape %s, creating projection with shape %s" % (x.shape, newx.shape)

        if randommatrix is None:
            print >> sys.stderr, "Creating random matrix of shape %s" % `(x.shape[1], options.dimensions)`
            print >> sys.stderr, stats()
            numpy.random.seed(options.seed)
            randommatrix = numpy.random.normal(size=(x.shape[1], options.dimensions))
        else:
            assert randommatrix.shape == (x.shape[1], options.dimensions)       # We assume the projection matrix won't change

        # Batch (cached, high-memory) random projection
        print >> sys.stderr, "Multiplying x by random matrix..."
        print >> sys.stderr, stats()
        newx = numpy.dot(x, randommatrix)
        print >> sys.stderr, "...done multiplying x by random matrix"
        print >> sys.stderr, stats()

        # Online (low-memory) random projection
#        # Iterate over every entry j,k that is nonzero in x
##        nonzeros = [idx.tolist() for idx in x.nonzero()]
##        for j, k in zip(nonzeros):
#        nonzeros = x.nonzero()
#        for l in range(len(nonzeros[0])):
#            j, k = nonzeros[0][l], nonzeros[1][l]
#        nonzero_pairs = [(nonzeros[0][l], nonzeros[1][l]) for l in range(len(nonzeros[0]))]
#
#            newrow = x[j,k] * pyrandomprojection.randomrow(key=k, dimensions=options.dimensions, RANDOM_SEED=options.seed)
#            assert newx[j].shape == newrow.shape
#            newx[j] += newrow
#
#            if l % 1000 == 999:
#                print >> sys.stderr, "Done with %s of nonzeroes on %s..." % (percent(l+1, len(nonzeros[0])), f)
#                print >> sys.stderr, stats()
