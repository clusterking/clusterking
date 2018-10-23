#!/usr/bin/env python3

from .inputs import inputs
from .form_factors import fplus, fzero, fT
import numpy as np

# A1


#  kinematic variables
#  q2,  El ,   thetal


#  thetal [0,pi]

#  q2 [   inputs['mtau']^2,  (inputs['mB'] -inputs['mD'])^2 ]

#  El  [     inputs['mtau']^2/(  2 * sqrt(q2) ),     sqrt(q2)/2   ]   w1   and [  0,  inputs['mtau']^2/(2 sqrt(q2))  ]  w2


def Klambda(a, b, c):
    return a ** 2 + b ** 2 + c ** 2 - 2 * (a * b + a * c + b * c)


def kvec(q2):
    return 1 / (2 * inputs['mB']) * np.sqrt(Klambda(inputs['mB'] ** 2, inputs['mD'] ** 2, q2))


##  23

def H0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (1 + epsL + epsR) * 2 * inputs['mB'] * kvec(q2) / np.sqrt(q2) * fplus(q2)


def Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (1 + epsL + epsR) * (inputs['mB'] ** 2 - inputs['mD'] ** 2) / (np.sqrt(q2)) * fzero(q2)


def HS(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return (epsSR + epsSL) * (inputs['mB'] ** 2 - inputs['mD'] ** 2) / (inputs['mb'] - inputs['mc']) * fzero(q2)


##

def Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return epsT * (2j * inputs['mB'] * kvec(q2)) / (inputs['mB'] + inputs['mD']) * fT(q2)


##  

def H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return epsT * (2j * inputs['mB'] * kvec(q2)) / (inputs['mB'] + inputs['mD']) * fT(q2)


#    32


def Icalzero(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return inputs['mtau'] * np.sqrt(q2) * np.absolute(H0val) ** 2 + 4 * inputs['mtau'] * np.sqrt(
        q2) * np.absolute(
        Hpmval + H0tval) ** 2 + 2j * inputs['mtau'] ** 2 * H0val * np.conjugate(
        Hpmval + H0tval) - 2j * q2 * np.conjugate(H0val) * (Hpmval + H0tval)


def IcalzeroI(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return - np.sqrt(q2) * np.conjugate(H0val) * (inputs['mtau'] * Htval + np.sqrt(q2) * HSval) \
           - 2j * inputs['mtau'] * np.conjugate(Hpmval + H0tval) * (inputs['mtau'] * Htval + np.sqrt(q2) * HSval)


Icalp = 0
Icalm = 0


## 30


def Gamma00p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(2j * np.sqrt(q2) * (Hpmval + H0tval) - inputs['mtau'] * H0val) ** 2


def Gammat0p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    # H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(inputs['mtau'] * Htval + np.sqrt(q2) * HSval) ** 2


def GammaI0p(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return 2 * np.real((2j * np.sqrt(q2) * (Hpmval + H0tval) - inputs['mtau'] * H0val) *
                       np.conjugate(inputs['mtau'] * Htval + np.sqrt(q2) * HSval))


Gammapp = 0

Gammamp = 0


def Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    H0val = H0(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    Hpmval = Hpm(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    H0tval = H0t(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # HSval = HS(epsL, epsR, epsSR, epsSL, epsT, q2, El)
    # Htval = Ht(epsL, epsR, epsSR, epsSL, epsT, q2, El)

    return np.absolute(np.sqrt(q2) * H0val - 2j * inputs['mtau'] * (Hpmval + H0tval)) ** 2


Gammapm = 0

Gammamm = 0


#    A2  and A3, A4


def Ical0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * np.real(2 * IcalzeroI(epsL, epsR, epsSR, epsSL, epsT, q2, El))


def Ical1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * np.real(2 * Icalzero(epsL, epsR, epsSR, epsSL, epsT, q2, El))


def Gammap0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * Gammat0p(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def Gammam0(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def Gammam2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return -2 * Gamma0m(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def Gammap2(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * Gamma00p(epsL, epsR, epsSR, epsSL, epsT, q2, El)


def Gammam1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 0.


def Gammap1(epsL, epsR, epsSR, epsSL, epsT, q2, El):
    return 2 * GammaI0p(epsL, epsR, epsSR, epsSL, epsT, q2, El)
