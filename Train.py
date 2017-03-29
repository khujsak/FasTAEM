# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:38:41 2017

Creates Training Dataset 

Variables defined in the beginning
Image defined second


@author: tomosys
"""

#Load in Standard Python Libraries
import numpy as np
from skimage.restoration import inpaint
import random
from random import randint
import time
import os
import cv2


#Load in Custom Functions
from dynsamp import imshow_pair, gauss_kern, padding_for_kernel, add_padding, remove_padding, distance_calc, get_samplereward 



start = time.time()
print("PreProcessing")

#Construct Image of Dimension Sc (Reshape)

sc=256

#Sample this many points randomly

p=randint(0,6000)

points=p+2000

#points=8000

#Dimensions of the Kernel 2* the Value in pixels

ksize=3

#Constant in the SD value for kernel

c=0.2

#Number of nearest measurements to keep in SD Value
L=6

#Need to define radius of z6 as a function of image area

rad=.2*sc


####Read in and Resize Image

os.chdir('/home/tomosys/Documents/SLADS/')
im = cv2.imread('100.bmp',0)

im=np.rint(im/44)

im=im[120:120+sc, 120:120+sc]


sz=np.shape(im)

imr=np.ravel(im)


ind=imr.size


#Construct M Training Datasets


for m in range(0,20):
   
    #Nomenclature
    #sp=sampled points in raveled
    #spv=sampled values raveled
    #usp=unsampledpoints in raveled
    #uspv=unsampled points raveled
    
    sp=np.array(random.sample(range(ind), points))
    
    spv=imr[sp]
    
    unsbool = np.ones(ind, dtype=bool)
    
    unsbool[sp]=False
    
    usp=np.where(unsbool)
    
    
    
    
    #Construct unsampled image
    
    ims=np.zeros(ind).astype(np.uint8)
    
    ims[sp]=imr[sp]
    
    
    #Inpaint Unsampled Image
    
    imsr=np.reshape(ims, sz)
    mask=np.ones(ind).astype(np.uint8)
    mask[sp]=0
    mask=np.reshape(mask, sz)
    
    #image_result = inpaint.inpaint_biharmonic(imsr, mask, multichannel=False)
    
    
    
    #dst = cv2.inpaint(imsr,dummat,6,cv2.INPAINT_TELEA)
    
    image_result = cv2.inpaint(imsr,mask,6,cv2.INPAINT_NS)
    
    imrr=np.ravel(image_result)
    
    uspv=imrr[usp]
    
    #
    #plt.figure()
    #plt.imshow(image_result, interpolation='none')
    #plt.figure()
    #plt.imshow(im, interpolation='none')
    #plt.figure()
    #plt.imshow(imsr, interpolation='none')
    
    #Inpaint ims
    
    #Compute Heurestic for Reduction in Distortion
    #Create Gaussian weighting function
    #Multiply times original image at location of 
    #Measured pixel, then sum up matrix
    
    #multiply a gaussian kernel through the distortion matrix 
    #and sum
    
    
    #Also compute sigma for each entry (distance
    #between nearest measured pixle)
    
    
    

    upiu=np.array(np.unravel_index(usp, [sc,sc]))
    
    #for each entry in upiu, calculate distance to all spiu
    #and take min
    
    rr=np.shape(upiu)[2]
    
    
    end = time.time()
    print(end - start)
    
    
    updis=np.zeros(rr)
    z3=np.zeros(rr)
    z4=np.zeros(rr)
    
    start = time.time()
    print("DistanceCalc")
    
    uspv=np.reshape(uspv, [sc*sc-points, 1])
    updis, z3, z4, z6=distance_calc(rr, usp, sp, spv, uspv, rad, L, disbig, sc, points, imr)
        
    
    
    
    #Extract image patches centered on the unsampled indices
    #Multiply by gaussian kernel and average
    
    dort=(np.abs(np.reshape(imr, [sc,sc])-image_result))
    
    
    
    end = time.time()
    print(end - start)
    
    
    h=gauss_kern(ksize, np.max(updis), c)
    
    
    i_pad, j_pad = padding_for_kernel(h)
    
    
    #Upiupad is only used during training, it's the padded indices
    
    upiupad=upiu
    
    upiupad[0,:]=np.add(upiupad[0,:],i_pad)
    upiupad[1,:]=np.add(upiupad[1,:],j_pad)
    
    dortpad=add_padding(dort, h)
    
    
    start = time.time()
    print("ValueCalc")
    
    
    Value=get_samplereward(upiupad, rr, updis, c, dortpad, ksize)
    
    
    end = time.time()
    print(end - start)
    
    
    
    
    
    
    #Compute a Feature Vector For Each Unsampled Point
    
    gx = np.gradient(image_result.astype("float"), axis=0)
    gy = np.gradient(image_result.astype("float"), axis=1)
    
    gxr=np.abs(np.ravel(gx))
    gyr=np.abs(np.ravel(gy))
    
    z1=gxr[usp]
    z2=gyr[usp]
    
    z5=updis
    
    
    Z=[z1, z2, z3, z4, z5, z6]
    
    for i in range(6):
        
        
        for j in range(6):
        
            
            if i>j:
                print('skip'+str(i)+str(j))
            else:
                dummy=Z[i][:] * Z[j][:]
                Z.append(dummy)
    
    
    
    Z=np.array(Z)
    
    Z=np.append(np.ones([1,sc*sc-points]), Z, 0)
    Z=np.transpose(Z)
    
    
    np.save('Z' + str(m) + '.npy', Z)
    np.save('V' + str(m) +'.npy', Value)
    #Construct Full Feature Vector
    
