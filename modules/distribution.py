#!/usr/bin/env python

import numpy as np

import scipy.integrate as integrate
from .amplitude import *
from .inputs import inputs

## some definitions

#   thetalmax =  np.pi

#   thetalmin = 0.


# cosine of the angle

cthetalmin = -1
cthetalmax = 1


q2min = inputs['mtau']**2
q2max = (inputs['mB'] - inputs['mD'])**2


def Elmax(q2):
    return np.sqrt(q2)/2.


Elmin = 0.

#  A1

def xval(q2):
    return np.sqrt(q2/inputs['mtau']**2)


def yval(El):
    return El/inputs['mtau']


def I0w1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap0val = Gammap0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam0val = Gammam0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam2val = Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammap2val = Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    Ical1val = Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return ((3*x**2 - 2)*(x + 4*y)*(x - 2*y)**2)/(6*x**2*(x**2 -1)**2 *y**2) * Gammap0val \
           + (2*x**4 - 3*x**2 -16*x*y**3 +12*y**2)/(6*x*(x**2-1)**2 *y**2) * Gammam0val \
           + (20*x**5 *y + x**4 *(40*y**2-6) +16*x**3*y*(5*y**2-4) +x**2*(15-72*y**2) -4*x*y*(8*y**2-5) +20*y**2)*(x-2*y)**2/(120*x*(x**2-1)**4*y**4)*Gammam2val \
           + (40*x**5*y +5*x**4*(16*y**2-3) - 50*x**3*y + x**2*(6-80*y**2) +16*x*y +24*y**2)*(x-2*y)**3/(120*x**2*(x**2-1)**4*y**4)*Gammap2val \
           + (-240*x**5*y**4 +9*x**5 +32*(10*x**4 -5*x**2 +1)*y**5 -30*(x**2+1)*x**4*y +20*(x**4+4*x**2+1)*x**3*y**2)/(120*x*(x**2-1)**4*y**4)*Ical1val


def I1w1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap1val = Gammap1(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam1val = Gammam1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    Ical0val = Ical0(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return (-2*x**4 + x**2 +4*(3*x**4 - 3*x**2 +1)*y**2 
            + (3*x**4 -5*x**2 +2)*x*y)*(x-2*y)**2/(6*x**2*(x**2-1)**3*y**3)*Gammap1val \
            + (2*x**6*y - x**5 - 3*x**4*y +x**3*(2-16*y**4) +x**2*y*(20*y**2-3)  -4*y**3)/(6*x*(x**2-1)**3*y**3)*Gammam1val \
            - ((x-2*y)**2*(2*x**3*y +x**2*(8*y**2-1)  -2*x*y -4*y**2))/(6*x *(x**2-1)**3*y**3)*Ical0val


def I2w1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap2val = Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam2val = Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    Ical1val = Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return  1/(120*(x**2-1)**4*y**4)*(720*x**3*y**4  -64*(5*(x**4+x**2)-1)*y**5  -60*x**2 *(x**4-2*x**2-2)*y +9*x**3*(2*x**2-5) +20*x*(2*x**6 -x**4 -16*x**2 -3)*y**2)*Gammam2val \
            + 1/(120*x**2*(x**2-1)**4*y**4)*(-720*x**7*y**4 +9*(5*x**2-2)*x**5  -60*(2*(x**4+x**2)-1)*x**4*y +64*(5*(3*x**6  -2*x**4 +x**2)-1)*y**5 +20*(3*x**6 +16*x**4 +x**2 -2)*x**3*y**2)*Gammap2val \
            + (240*x**5*y**4  -9*x**5  -32*(10*x**4 -5*x**2 +1)*y**5 +30*(x**2+1)*x**4*y  -20*(x**4 +4*x**2 +1)*x**3*y**2)/(40*x*(x**2-1)**4 *y**4)*Ical1val


def I0w2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap0val = Gammap0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam0val = Gammam0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammap2val = Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam2val = Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El)


    #  Ical0val = Ical0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Ical1val = Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return - 2*(2*x**2+1)*(4*x*y-3)/(3*x)*Gammam0val \
           + 2*(x**2+2)*(3*x-4*y)/(3*x**2)*Gammap0val \
           + 2/15*(-12*x**2*y +10*x +5/x -8*y)*Gammam2val \
           + (10*x*(x**2+2) - 8*(2*x**2+3)*y)/(15*x**2)*Gammap2val \
           - 4*(x**2-1)*y/(15*x)*Ical1val


def I1w2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap1val = Gammap1(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam1val = Gammam1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    Ical0val = Ical0(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return (8*x**3*y - 4*x**2+2)/(3*x)*Gammam1val \
           - 2*(x**3 - 2*x + 4*y)/(3*x**2)*Gammap1val \
           + 4/3*(-2*x*y - 2*y/x +1)*Ical0val


def I2w2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    x = xval(q2)
    y = yval(El)

    Gammap2val = Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Gammam2val = Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    Ical1val = Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 8*(x**2-1)*y/(15*x**2)*Gammap2val \
           - 8/15*(x**2-1)*y*Gammam2val \
           + 4*(x**2-1)*y/(5*x)*Ical1val


def I0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    if inputs['mtau']**2/(2 * np.sqrt(q2)) <= El <= np.sqrt(q2)/2.:
        return I0w1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    # todo: shouldn't that just be an else statement?
    elif inputs['mtau']**2/(2 * np.sqrt(q2)) > El >= 0:
        return I0w2(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def I1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    if inputs['mtau']**2/(2 *np.sqrt(q2)) <= El <= np.sqrt(q2)/2.:
        return I1w1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    # todo: shouldn't that just be an else statement?
    elif inputs['mtau']**2/(2 * np.sqrt(q2)) > El >= 0:
        return I1w2(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def I2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    if inputs['mtau']**2/(2 * np.sqrt(q2)) <= El <= np.sqrt(q2)/2.:
        return I2w1(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    # todo: shouldn't that just be an else statement?
    elif inputs['mtau']**2/(2 * np.sqrt(q2)) > El >= 0:
        return I2w2(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def dG(epsL, epsR, epsSR, epsSL, epsT, q2, El, cthetal):
    """3D diff. distribution over q2, El  and cos(thetal) """
    I0val = I0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    I1val = I1(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    I2val = I2(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return inputs['Btaul'] * inputs['GF']**2 * np.absolute(inputs['Vcb'])**2 * inputs['new']**2/(32 * np.pi**3) * kvec(q2)/inputs['mB']**2 * (1-inputs['mtau']**2/q2)**2 * El**2/inputs['mtau']**3 *(I0val + I1val * cthetal + I2val * cthetal**2)

def dGq2El(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    I0val = I0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    I2val = I2(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return inputs['Btaul'] * inputs['GF']**2 * np.absolute(inputs['Vcb'])**2 * inputs['new']**2/(16 * np.pi**3) * kvec(q2)/inputs['mB']**2 * (1-inputs['mtau']**2/q2)**2 * El**2/inputs['mtau']**3 *(I0val + 1./3. * I2val)


def dGq2(epsL, epsR, epsSR, epsSL, epsT, q2):
    """ 1D q2 distrubution, integrating El """

    return integrate.quad(
        lambda El: dGq2El(epsL, epsR, epsSR, epsSL, epsT, q2, El),
        Elmin,
        Elmax(q2))[0]



def dGq2norm(epsL, epsR, epsSR, epsSL, epsT, q2):
    """Normalized distribution 1D q2 """

    return inputs['tauBp']/inputs['Btaul'] * dGq2(epsL, epsR, epsSR, epsSL, epsT, q2)


def q2inflim(El):
    """Define lower limit of integration for q2, see Fig 4 in Alonso et al
    paper
    """

    if El > inputs['mtau']/2:
        return 4*El**2

    else:
        return q2min


# todo: do not ignore second output of integrate.quad (errors/warnings)?


def dGEl(epsL, epsR, epsSR, epsSL, epsT, El):
    """1D El distrubution, integrating over q2"""

    return integrate.quad(
        lambda q2: dGq2El(epsL, epsR, epsSR, epsSL, epsT, q2, El),
        q2inflim(El),
        q2max)[0]


def dGElnorm(epsL, epsR, epsSR, epsSL, epsT, El):
    """Normalized distribution 1D El """
    return inputs['tauBp']/inputs['Btaul'] * dGEl(epsL, epsR, epsSR, epsSL, epsT, El)


def dGq2cthetal(epsL, epsR, epsSR, epsSL, epsT, q2, cthetal):
    """2D  q2- cthetal distrubution, integrate El"""
    return integrate.quad(
         lambda El: dG(epsL, epsR, epsSR, epsSL, epsT, q2, El, cthetal),
         Elmin,
         Elmax(q2))[0]


def dGcthetal(epsL, epsR, epsSR, epsSL, epsT, cthetal):
    """1D  cthetal distrubution, integrate q2"""

    return integrate.quad(
        lambda q2: dGq2cthetal(epsL, epsR, epsSR, epsSL, epsT, q2, cthetal),
        q2min,
        q2max)[0]


def dGcthetalnorm(epsL, epsR, epsSR, epsSL, epsT, cthetal):
    """Normalized distribution 1D cthetal"""

    return inputs['tauBp']/inputs['Btaul'] * dGcthetal(epsL, epsR, epsSR, epsSL, epsT,cthetal)


def dGtot(epsL, epsR, epsSR, epsSL, epsT):
    """Total decay rate"""

    return integrate.quad(
        lambda q2: dGq2(epsL, epsR, epsSR, epsSL, epsT, q2),
        q2min,
        q2max)[0]


def dGq2normtot(epsL, epsR, epsSR, epsSL, epsT, q2):
    """q2 distribution normalized by total,  integral of this would be 1
    by definition
    """

    return dGq2(epsL, epsR, epsSR, epsSL, epsT, q2)/\
           dGtot(epsL, epsR, epsSR, epsSL, epsT)


def dGElnormtot(epsL, epsR, epsSR, epsSL, epsT, El):
    """El distribution normalized by total,  integral of this would be 1 by
    definition
    """

    return dGEl(epsL, epsR, epsSR, epsSL, epsT, El)/\
           dGtot(epsL, epsR, epsSR, epsSL, epsT)


def dGcthetalnormtot(epsL, epsR, epsSR, epsSL, epsT, cthetal):
    """cthetal distribution normalized by total,  integral of this would be 1
    by definition
    """

    return dGcthetal(epsL, epsR, epsSR, epsSL, epsT, cthetal)/\
           dGtot(epsL, epsR, epsSR, epsSL, epsT)


def dGnormtot(epsL, epsR, epsSR, epsSL, epsT, q2, El, cthetal):
    """Full 3D distribution normalized by total,  3D integral of this would be
    1 by definition
    """

    return dG(epsL, epsR, epsSR, epsSL, epsT, q2, El , cthetal)/\
           dGtot(epsL, epsR, epsSR, epsSL, epsT)
