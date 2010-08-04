#!/usr/bin/env python
"""
Import unlabelled data (pkl files containing a numpy array), and perform a random projection on it.
Save to a corresponding .pkl file.

USAGE:
    ./randomprojection.py [-d dimensions] file1.pkl ...

    e.g. ./randomprojection.py -d 1000 /u/glorotxa/work/NLP/DARPAproject/*instances*pkl

NOTE:
    * Right now, we only use a Gaussian random matrix.

REQUIREMENTS:
    * http://github.com/turian/pyrandomprojection
    * http://github.com/turian/common
    * NumPy
"""

import pyrandomprojection
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
    (options, args) = parser.parse_args()

    assert len(args) > 0    # You need to pass in pkl files to project.

    for i, f in enumerate(args):
        print >> sys.stderr, "\nLoading %s (file %s)..." % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()
        x = cPickle.load(open(f, "rb"))
        print >> sys.stderr, "...loading %s (file %s)" % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()

        assert x.ndim == 2
        newx = numpy.zeros((x.shape[0], options.dimensions))
        print >> sys.stderr, "Read instance matrix with shape %s, creating projection with shape %s" % (x.shape, newx.shape)

        # Iterate over every entry j,k that is nonzero in x
#        nonzeros = [idx.tolist() for idx in x.nonzero()]
#        for j, k in zip(nonzeros):
        nonzeros = x.nonzero()
        for l in range(len(nonzeros[0])):
            j, k = nonzeros[0][l], nonzeros[1][l]

            newrow = x[j,k] * pyrandomprojection.randomrow(key=(j,k), dimensions=options.dimensions)
            assert newx[j].shape == newrow.shape
            newx[j] += newrow

            if l % 1000 == 999:
                print >> sys.stderr, "Done with %s of nonzeroes on %s..." % (percent(l+1, len(nonzeros[0])), f)
                print >> sys.stderr, stats()
                break
