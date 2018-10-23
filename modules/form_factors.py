#!/usr/bin/env python3

from flavio.physics.bdecays.formfactors.b_p import bcl
from flavio.parameters import default_parameters

default_central_values = default_parameters.get_central_all()


def fplus(q2):
    return bcl.ff_isgurwise('B->D', q2, default_central_values, 4.8, n=3)['f+']
    

def fzero(q2):
    return bcl.ff_isgurwise('B->D', q2, default_central_values, 4.8, n=3)['f0']


def fT(q2):
    return bcl.ff_isgurwise('B->D', q2, default_central_values, 4.8, n=3)['fT']
