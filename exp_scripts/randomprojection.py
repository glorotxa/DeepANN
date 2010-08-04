#!/usr/bin/env python
"""
Import unlabelled data (pkl files containing a numpy bool array), and perform a random projection on it.
Save to a corresponding .pkl file.

USAGE:
    ./randomprojection.py [-d dimensions] file1.pkl ...

    e.g. ./randomprojection.py -d 1000 /u/glorotxa/work/NLP/DARPAproject/*pkl

NOTE:
    * Right now, we only use a Gaussian random matrix.

REQUIREMENTS:
    * http://github.com/turian/pyrandomprojection
    * http://github.com/turian/common
"""

import pyrandomprojection
from common.stats import stats
from common.str import percent

import sys
import cPickle

#def project(dict, dimensions, RANDOMIZATION_TYPE="gaussian", TERNARY_NON_ZERO_PERCENT=None, RANDOM_SEED=0):

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-d", "--dimensions", dest="dimensions", default=500, type="int", help="number of dimensions in random output")
    (options, args) = parser.parse_args()

    assert len(args) > 0    # You need to pass in pkl files to project.

    for i, f in enumerate(args):
        print >> sys.stderr, "\nLoading %s (file %s)..." % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()
        x = cPickle.load(open(f, "rb"))
        print >> sys.stderr, "...loading %s (file %s)" % (f, percent(i+1, len(args)))
        print >> sys.stderr, stats()
