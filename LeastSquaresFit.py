# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:50:59 2017

@author: tomosys


Loads in all the feature vectors calculated previously and performs a 
least squares fit to estimate the values of the coefficients.  These 
coefficients can then be used for prediction.
"""
import numpy as np

subjects=range(1,5)

Z=np.array([])
for subject in subjects:
    f=np.load('Z%d.npy' % (subject))
    
    if subject==1:
        Z=f
    else:
        Z=np.append(Z,f, axis=0)

V=np.array([])        
for subject in subjects:
    f=np.load('V%d.npy' % (subject))
    
    if subject==1:
        V=f
    else:
        V=np.append(V,f, axis=0)
        
        
coeff=np.linalg.lstsq(Z, V)
