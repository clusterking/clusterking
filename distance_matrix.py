#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





df=pd.read_csv('./global_results.out',sep='\s+',header=None,names= ["epsL","epsSL","epsT","q2","dist"])



nt = 15





for j in range(1,8001):
    for i in range(j,8001):
            df1 = df[(i-1)*nt:(i)*nt].reset_index()
            df2 = df[(j-1)*nt:(j)*nt].reset_index()
            chi2 = sum((df1.dist -df2.dist )**2)
            print(i,j,chi2)

