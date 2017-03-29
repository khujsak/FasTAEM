# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:51:16 2017

@author: tomosys
"""
import numpy as np

disbig=np.zeros([65536,65536]).astype(np.float16)

for i in range(256):
    for j in range(256):
        
        n=256
        m=256
        
        xt=np.linspace(0,255,256)
        yt=np.linspace(0,255,256)
        
        
        
        
        xv, yv = np.meshgrid(xt, yt)
        
        dist=np.sqrt((xv-j)**2 + (yv-i)**2)
        
        dist=dist.astype(np.float32)
        
        disbig[256*i+j, :]=np.ravel(dist)