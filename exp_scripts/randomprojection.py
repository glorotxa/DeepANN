#!/usr/bin/env python
"""
Import unlabelled data (pkl files containing a numpy array), and perform a random projection on it.
Save to a corresponding .pkl file.

USAGE:
    ./randomprojection.py [-d dimensions] [-s seed] file1.pkl ...

    e.g. ./randomprojection.py -d 1000 /u/glorotxa/work/NLP/DARPAproject/*instances*pkl

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

MODE = "online"     # Deterministic, low-memory version
#MODE = "batch"     # Cache the entire random matrix in memory. (However,
                    # this might not be deterministic. So try to do all
                    # your random projections in one invocation.)
                    # Could be, in principle, slower than online if the
                    # number of entries >> #nonzeros.

if MODE == "online":
    import pyrandomprojection


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

    if MODE == "batch":
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

        if MODE == "batch":
            # Batch (cached, high-memory) random projection
            if randommatrix is None:
                print >> sys.stderr, "Creating random matrix of shape %s" % `(x.shape[1], options.dimensions)`
                print >> sys.stderr, stats()
                numpy.random.seed(options.seed)
                randommatrix = numpy.random.normal(size=(x.shape[1], options.dimensions))
            else:
                assert randommatrix.shape == (x.shape[1], options.dimensions)       # We assume the projection matrix won't change

            print >> sys.stderr, "Multiplying x by random matrix..."
            print >> sys.stderr, stats()
            newx = numpy.dot(x, randommatrix)
            print >> sys.stderr, "...done multiplying x by random matrix"
            print >> sys.stderr, stats()

        elif MODE == "online":
            # Online (low-memory) random projection
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
                    randrow_values = pyrandomprojection.randomrow(key=col, dimensions=options.dimensions, RANDOM_SEED=options.seed)

                    randrows_computed += 1
                    if randrows_computed % 100 == 0:
                        print >> sys.stderr, "Retrieved %s random rows thus far, done with %s of nonzeroes on %s..." % (percent(randrows_computed, x.shape[1]), percent(l+1, len(nonzero_colrow)), f)
                        print >> sys.stderr, stats()
                newrow = x[row,col] * randrow_values
                assert newx[row].shape == newrow.shape
                newx[row] += newrow
    
#                if (l+1) % 10000 == 0:
#                    print >> sys.stderr, "Done with %s of nonzeroes on %s..." % (percent(l+1, len(nonzero_colrow)), f)
#                    print >> sys.stderr, stats()

        else: assert 0
