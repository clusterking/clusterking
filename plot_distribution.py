#!/usr/bin/env python3

from modules.distribution import bin_function, dGq2, q2min, q2max
import numpy as np
from modules.inputs import Wilson

import matplotlib.pyplot as plt

if __name__ == "__main__":
    w = Wilson(0, 0, 0, 0, 0)
    print("Calculating")
    values = bin_function(lambda x: dGq2(w, x),
                          np.linspace(q2min, q2max, 10),
                          normalized=True,
                          midpoints=True)
    values_fine = bin_function(lambda x: dGq2(w, x),
                          np.linspace(q2min, q2max, 20),
                          normalized=True,
                          midpoints=True)
    print("Plotting")
    plt.step(*zip(*values))
    plt.step(*zip(*values_fine))
    plt.show()