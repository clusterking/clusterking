#!/usr/bin/env python

import numpy as np

import flavio
from math import sqrt,radians,asin
from flavio.physics.bdecays.formfactors.b_p import btop, bcl_parameters, bcl
import copy
from flavio.parameters import default_parameters
from flavio.classes import Parameter, Implementation



c = copy.deepcopy(default_parameters)
par = c.get_central_all()




#  {'f+','f0','fT'}


def fplus(q2):
    
    fflist = list(bcl.ff_isgurwise('B->D',q2, par, 4.8, n=3).values())
    
    return  fflist[0]



def fzero(q2):
    
    fflist = list(bcl.ff_isgurwise('B->D',q2, par, 4.8, n=3).values())
    
    return  fflist[1]




def fT(q2):
    
    fflist = list(bcl.ff_isgurwise('B->D',q2, par, 4.8, n=3).values())
    
    return  fflist[2]



#print( fplus(4),fzero(4),fT(4)  )

#print(bcl.ff_isgurwise('B->D', 4, par, 4.8, n=3))
