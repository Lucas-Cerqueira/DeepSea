import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

from RBM import RBM as rbmLib

print 'Creating RBM'
rbm = rbmLib.RBM(input=None,n_visible = 10,n_hidden=10,W=None,hbias=None,vbias=None,numpy_rng=None,theano_rng=None)

print rbm