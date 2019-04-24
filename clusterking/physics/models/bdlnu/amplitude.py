#!/usr/bin/env python3

# std
import numpy as np

# 3rd party
from wilson import Wilson

# ours
from clusterking.physics.models.bdlnu.form_factors import fplus, fzero, fT
from clusterking.physics.models.bdlnu.inputs import inputs

# todo: make pycharm ignore name convention pylinting in this file

#  kinematic variables
#  q2,  El ,   thetal

## Limits:

#  thetal [0,pi]

#  q2 [   inputs['mtau']^2,  (inputs['mB'] -inputs['mD'])^2 ]

#  El  [     inputs['mtau']^2/(  2 * sqrt(q2) ),     sqrt(q2)/2   ] for  w1   and [  0,  inputs['mtau']^2/(2 sqrt(q2))  ]  for w2


def Klambda(a, b, c):
    # even though the formula is positive definite, use max to enforce this
    # even when rounding errors occurr (else problems with sqrt later)
    return max(0 , a ** 2 + b ** 2 + c ** 2 - 2 * (a * b + a * c + b * c))


def kvec(q2):
    return 1 / (2 * inputs['mB']) * np.sqrt(Klambda(inputs['mB'] ** 2, inputs['mD'] ** 2, q2))


##  23

def H0(w: Wilson, q2, El):
    return (1 + w.match_run(inputs["mb"], "WET", "flavio")["CVL_bctaunutau"] + w.match_run(inputs["mb"], "WET", "flavio")["CVR_bctaunutau"]) * 2 * inputs['mB'] * kvec(q2) / np.sqrt(q2) * fplus(q2)


def Ht(w: Wilson, q2, El):
    return (1 + w.match_run(inputs["mb"], "WET", "flavio")["CVL_bctaunutau"] + w.match_run(inputs["mb"], "WET", "flavio")["CVR_bctaunutau"]) * (inputs['mB'] ** 2 - inputs['mD'] ** 2) / (np.sqrt(q2)) * fzero(q2)


def HS(w: Wilson, q2, El):
    return (w.match_run(inputs["mb"], "WET", "flavio")["CSR_bctaunutau"] + w.match_run(inputs["mb"], "WET", "flavio")["CSL_bctaunutau"]) * (inputs['mB'] ** 2 - inputs['mD'] ** 2) / (inputs['mb'] - inputs['mc']) * fzero(q2)


##

def Hpm(w: Wilson, q2, El):
    return w.match_run(inputs["mb"], "WET", "flavio")["CT_bctaunutau"] * (2j * inputs['mB'] * kvec(q2)) / (inputs['mB'] + inputs['mD']) * fT(q2)


##

def H0t(w: Wilson, q2, El):
    return w.match_run(inputs["mb"], "WET", "flavio")["CT_bctaunutau"] * (2j * inputs['mB'] * kvec(q2)) / (inputs['mB'] + inputs['mD']) * fT(q2)


#    32


def Icalzero(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)

    return inputs['mtau'] * np.sqrt(q2) * np.absolute(H0val) ** 2 + 4 * inputs['mtau'] * np.sqrt(
        q2) * np.absolute(
        Hpmval + H0tval) ** 2 + 2j * inputs['mtau'] ** 2 * H0val * np.conjugate(
        Hpmval + H0tval) - 2j * q2 * np.conjugate(H0val) * (Hpmval + H0tval)


def IcalzeroI(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return - np.sqrt(q2) * np.conjugate(H0val) * (inputs['mtau'] * Htval + np.sqrt(q2) * HSval) \
           - 2j * inputs['mtau'] * np.conjugate(Hpmval + H0tval) * (inputs['mtau'] * Htval + np.sqrt(q2) * HSval)


Icalp = 0
Icalm = 0


## 30


def Gamma00p(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    # HSval = HS(w, q2, El)

    return np.absolute(2j * np.sqrt(q2) * (Hpmval + H0tval) - inputs['mtau'] * H0val) ** 2


def Gammat0p(w: Wilson, q2, El):
    # H0val = H0(w, q2, El)
    # Hpmval = Hpm(w, q2, El)
    # H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return np.absolute(inputs['mtau'] * Htval + np.sqrt(q2) * HSval) ** 2


def GammaI0p(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return 2 * np.real((2j * np.sqrt(q2) * (Hpmval + H0tval) - inputs['mtau'] * H0val) *
                       np.conjugate(inputs['mtau'] * Htval + np.sqrt(q2) * HSval))


Gammapp = 0

Gammamp = 0


def Gamma0m(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    # HSval = HS(w, q2, El)
    # Htval = Ht(w, q2, El)

    return np.absolute(np.sqrt(q2) * H0val - 2j * inputs['mtau'] * (Hpmval + H0tval)) ** 2


Gammapm = 0

Gammamm = 0


#    A2  and A3, A4


def Ical0(w: Wilson, q2, El):
    return 2 * np.real(2 * IcalzeroI(w, q2, El))


def Ical1(w: Wilson, q2, El):
    return 2 * np.real(2 * Icalzero(w, q2, El))


def Gammap0(w: Wilson, q2, El):
    return 2 * Gammat0p(w, q2, El)


def Gammam0(w: Wilson, q2, El):
    return 2 * Gamma0m(w, q2, El)


def Gammam2(w: Wilson, q2, El):
    return -2 * Gamma0m(w, q2, El)


def Gammap2(w: Wilson, q2, El):
    return 2 * Gamma00p(w, q2, El)


def Gammam1(w: Wilson, q2, El):
    return 0.


def Gammap1(w: Wilson, q2, El):
    return 2 * GammaI0p(w, q2, El)
