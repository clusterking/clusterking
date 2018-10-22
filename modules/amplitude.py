#!/usr/bin/env python3

from .inputs import *
from .form_factors import *


# A1



#  NP wilson coefficients
#  epsL, epsR, epsSR, epsSL, epsT


#  kinematic variables
#  q2,  El ,   thetal


#  thetal [0,pi]

#  q2 [   mtau^2,  (mB -mD)^2 ]

#  El  [     mtau^2/(  2 * sqrt(q2) ),     sqrt(q2)/2   ]   w1   and [  0,  mtau^2/(2 sqrt(q2))  ]  w2


def Klambda(a, b, c):
    return a ** 2 + b ** 2 + c ** 2 - 2 * (a * b + a * c + b * c)

def kvec(q2):
    return 1 / (2 * mB) * np.sqrt(Klambda(mB ** 2, mD ** 2, q2))


##  23

def H0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (1 + epsL + epsR) * 2 * mB * kvec(q2) / np.sqrt(q2) * fplus(q2)


def Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (1 + epsL + epsR) * (mB ** 2 - mD ** 2) / (np.sqrt(q2)) * fzero(q2)


def HS(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (epsSR + epsSL) * (mB ** 2 - mD ** 2) / (mb - mc) * fzero(q2)


##

def Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return epsT * (2j * mB * kvec(q2)) / (mB + mD) * fT(q2)


##  

def H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return epsT * (2j * mB * kvec(q2)) / (mB + mD) * fT(q2)


#    32


def Icalzero(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return mtau * np.sqrt(q2) * np.absolute(H0val) ** 2 + 4 * mtau * np.sqrt(
        q2) * np.absolute(
        Hpmval + H0tval) ** 2 + 2j * mtau ** 2 * H0val * np.conjugate(
        Hpmval + H0tval) - 2j * q2 * np.conjugate(H0val) * (Hpmval + H0tval)


def IcalzeroI(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return - np.sqrt(q2) * np.conjugate(H0val) * (
    mtau * Htval + np.sqrt(q2) * HSval) - 2j * mtau * np.conjugate(
        Hpmval + H0tval) * (mtau * Htval + np.sqrt(q2) * HSval)


Icalp = 0

Icalm = 0


## 30


def Gamma00p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(2j * np.sqrt(q2) * (Hpmval + H0tval) - mtau * H0val) ** 2


def Gammat0p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    # H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(mtau * Htval + np.sqrt(q2) * HSval) ** 2


def GammaI0p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * np.real((2j * np.sqrt(q2) * (Hpmval + H0tval) - mtau * H0val) *
                       np.conjugate(mtau * Htval + np.sqrt(q2) * HSval))


Gammapp = 0

Gammamp = 0


def Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(np.sqrt(q2) * H0val - 2j * mtau * (Hpmval + H0tval)) ** 2


Gammapm = 0

Gammamm = 0


#    A2  and A3, A4


def Ical0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    IcalzeroIval = IcalzeroI(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * np.real(2 * IcalzeroIval)


def Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    Icalzeroval = Icalzero(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * np.real(2 * Icalzeroval)


def Gammap0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    Gammat0pval = Gammat0p(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * Gammat0pval


def Gammam0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    Gamma0mval = Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * Gamma0mval


def Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    Gamma0mval = Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return -2 * Gamma0mval


def Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    Gamma00pval = Gamma00p(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * Gamma00pval


def Gammam1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 0.


def Gammap1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    GammaI0pval = GammaI0p(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * GammaI0pval
