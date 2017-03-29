# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:37:52 2017

@author: tomosys
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import inpaint

import scipy
import random

from random import randint

from numba import jit
import prox_tv as ptv

import time


import os
import cv2



def imshow_pair(image_pair, titles=('', ''), figsize=(10, 5), **kwargs):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    for ax, img, label in zip(axes.ravel(), image_pair, titles):
        ax.imshow(img, **kwargs)
        ax.set_title(label)

@jit
def gauss_kern(size, sigma, const):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    sizey = size
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-const*(x**2/float(size)+y**2/float(sizey))/(2*(sigma)**2))
    return g / g.max()


def padding_for_kernel(kernel):
    """ Return the amount of padding needed for each side of an image.

    For example, if the returned result is [1, 2], then this means an
    image should be padded with 1 extra row on top and bottom, and 2
    extra columns on the left and right.
    """
    # Slice to ignore RGB channels if they exist.
    image_shape = kernel.shape[:2]
    # We only handle kernels with odd dimensions so make sure that's true.
    # (The "center" pixel of an even number of pixels is arbitrary.)
    assert all((size % 2) == 1 for size in image_shape)
    return [(size - 1) // 2 for size in image_shape]


   
def add_padding(image, kernel):
    h_pad, w_pad = padding_for_kernel(kernel)
    return np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)),
                  mode='constant', constant_values=0)


def remove_padding(image, kernel):
    inner_region = []  # A 2D slice for grabbing the inner image region
    for pad in padding_for_kernel(kernel):
        slice_i = slice(None) if pad == 0 else slice(pad, -pad)
        inner_region.append(slice_i)
    return image[inner_region]



@jit
def window_slice(center, kernel):
    r, c = center
    r_pad, c_pad = padding_for_kernel(kernel)
    # Slicing is (inclusive, exclusive) so add 1 to the stop value
    return [slice(r-r_pad, r+r_pad+1), slice(c-c_pad, c+c_pad+1)]


@jit
def apply_kernel(center, kernel, original_image): 
    #Need a fix here to convert the coordinates on the original image
    #To the coordinates in the padded image    
    image_patch = original_image[window_slice(center, kernel)]
    
    # An element-wise multiplication followed by the sum
    return np.sum(np.multiply(kernel, image_patch))


@jit
def distance_calc(rr, unsampled, sp, sampledvals, unsampledvals, rad, L, disbig, sc, points, imr):
    """ Computes distance between an unsampled point and all the sampled points.
    The smallest distance is stored in updis (one for every unsampled point) and
    the L smallest distances are stored in gret and gret vals.  Gret has the 
    indices of the closest L values and Gretvals has their value in the image.
    These are used to calculate z3, z4, and z6, and the Value estimates."""
    
    indd=(np.size(unsampled))
    updis=np.zeros(indd)
    z3=np.zeros(indd)
    z4=np.zeros(indd)
    z6=np.ones(indd)
    
    
    z6=np.multiply(z6, 1/(1+np.pi*(rad**2)))
    
    
    gret=np.zeros([indd, L])
    gret=gret.astype(np.int)
    gretvals=np.zeros([indd, L])
    D34=np.zeros([indd, L])
    #wr=np.zeros(L)
    #updis=np.zeros(rr)
    
    
    
    
    for i in range(rr):
    #Calculate DIst
    
        #upiui=[upiu[0][0][i], upiu[1][0][i]]
        
        dist=disbig[unsampled[0][i], sp]
        #dist = np.sqrt(np.square(upiu[0][0][i]-spiu[0]) + np.square(upiu[1][0][i]-spiu[1]))
        
        
        #Calculate Z3, Z4
        gret[i,:]=(dist.argsort()[:L])
        #gret=np.argpartition(dist, 4)[:L]
        
        gretvals[i,:]=dist[gret[i,:]]
        
        updis[i]=gretvals[i,0]
        
        z6[i]=z6[i]*(1+np.sum(dist < rad))
        
        #Move all of the below out of the for loop and vectorize
        
        
    D34=np.abs(imr[gret].astype(np.float)-unsampledvals.astype(np.float))
    
    z3=np.sqrt((1/L)*np.sum(D34**2, 1))
    
    gretvsquare=gretvals**2
    
    #wr=(1/(gretvsquare)/np.sum(1/(gretvsquare)))
    
    wr=(1/(gretvsquare))/np.reshape(np.sum(1/gretvsquare, 1), [sc*sc-points,1])        
        
    z4=np.sum(wr*D34,1)
    
    #Need measured area
    #Rad is in terms of pixels
    
    
        
    
    return updis, z3, z4, z6



@jit
def get_samplereward(indexlist, rr, dist, c, imagepad, ksize):
    """ Computes the Expected Reduction in Distortion from sampling an
    unsampled pixel using the distance calculated above.  A Gaussian function
    is used to weight nearby unsampled pixels as well and the total difference
    between the ground truth and the inpainted weighted by this gaussian
    is used to estimate the value of sampling and inpainting.  A value is
    calculated for every unsampled pioxel and passed back to the main function.
    """
    index=np.zeros((1,2), dtype=np.int)
    h=np.zeros([ksize,ksize], dtype=np.float64)
    out=np.zeros(rr, dtype=np.float64)
    
    for i in range(rr):
        
        index=[indexlist[0][0][i], indexlist[1][0][i]]
        h=gauss_kern(ksize, dist[i], c)
    
    
        imagepad[window_slice(index, h)]
        
        out[i]=apply_kernel(index, h, imagepad)
    
    return out



""" Following Functions are Used for Prediction"""



def random_inpaint(imr, points, sz):
    
    ind=imr.size
    
    sp=np.array(random.sample(range(ind), points))
    ims=np.zeros(ind).astype(np.uint8)

    ims[sp]=imr[sp]
    
    
    #Inpaint Unsampled Image
    
    imsr=np.reshape(ims, sz)
    mask=np.ones(ind).astype(np.uint8)
    mask[sp]=0
    mask=np.reshape(mask, sz)
    
    #image_result = inpaint.inpaint_biharmonic(imsr, mask, multichannel=False)
    
    
    
    #dst = cv2.inpaint(imsr,dummat,6,cv2.INPAINT_TELEA)
    
    return cv2.inpaint(imsr,mask,6,cv2.INPAINT_NS)
    #return inpaint.inpaint_biharmonic(imsr, mask, multichannel=False)


def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
 
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")
 
	# show the images
	plt.show()


@jit
def distance_calc_pred(rr, unsampled, sp, sampledvals, unsampledvals, rad, L, disbig, sc, points, imr):

    indd=(np.size(unsampled))
    updis=np.zeros(indd)
    z3=np.zeros(indd)
    z4=np.zeros(indd)
    z6=np.ones(indd)
    
    
    z6=np.multiply(z6, 1/(1+np.pi*(rad**2)))
    
    
    gret=np.zeros([indd, L])
    gret=gret.astype(np.int)
    gretvals=np.zeros([indd, L])
    D34=np.zeros([indd, L])
    #wr=np.zeros(L)
    #updis=np.zeros(rr)
    
    
    
    
    for i in range(rr):
    #Calculate DIst
    
        #upiui=[upiu[0][0][i], upiu[1][0][i]]
        
        dist=disbig[unsampled[0][i], sp]
        #dist = np.sqrt(np.square(upiu[0][0][i]-spiu[0]) + np.square(upiu[1][0][i]-spiu[1]))
        
        
        #Calculate Z3, Z4
        gret[i,:]=(dist.argsort()[:L])
        #gret=np.argpartition(dist, 4)[:L]
        
        gretvals[i,:]=dist[gret[i,:]]
        
        updis[i]=gretvals[i,0]
        
        z6[i]=z6[i]*(1+np.sum(dist < rad))
        
        #Move all of the below out of the for loop and vectorize
        
    #unsampledvals=np.transpose(unsampledvals)     
    
    D34=np.abs(imr[gret].astype(np.float)-unsampledvals.astype(np.float))
    
    z3=np.sqrt((1/L)*np.sum(D34**2, 1))
    
    gretvsquare=gretvals**2
    
    #wr=(1/(gretvsquare)/np.sum(1/(gretvsquare)))
    
    wr=(1/(gretvsquare))/np.reshape(np.sum(1/gretvsquare, 1), [sc*sc-points,1])        
        
    z4=np.sum(wr*D34,1)
    
    #Need measured area
    #Rad is in terms of pixels
    
    
        
    
    return updis, z3, z4, z6, gret, gretvals


@jit
def update_featurevector(rr, unsampled, unsampledvals, rad, L, disbig, gret, gretvals, z6, imr, imsnew):
    
    
    
    #Then update values
    updis=gretvals[:, 0]    
    
    gretvalsns=np.zeros([rr,L])
    gretns=np.zeros([rr,L]).astype(np.int)
    distn=disbig[imsnew, unsampled]
    
    test=np.append(gretvals, np.reshape(distn, [rr,1]), axis=1)
    testarg=np.argsort(test, axis=1)
    
    gretn=np.append(gret, imsnew*np.ones([rr,1]), axis=1)  
    
    
    
    for i in range(rr):
        gretvalsns[i][:]=test[i,testarg[i,:]][:L]
        gretns[i][:]=gretn[i,testarg[i,:]][:L]

    distn=np.reshape(distn, [rr])
    tests=(updis>distn)
    updis[tests]=distn[tests]
    z5n=updis
    
    
    #z6n+=(distn*(distn<rad))/(1+np.pi*(rad**2))
    unsampledvals=np.transpose(unsampledvals)    
    
    D34n=np.abs(imr[gretns].astype(np.float)-unsampledvals.astype(np.float))
    
    z3n=np.sqrt((1/L)*np.sum(D34n**2, 1))
    
    gretvsquare=gretvalsns**2
    
    #wr=(1/(gretvsquare)/np.sum(1/(gretvsquare)))
    
    wr=(1/(gretvsquare))/np.reshape(np.sum(1/gretvsquare, 1), [rr,1])        
        
    z4n=np.sum(wr*D34n,1)
    
    z6n=z6+np.array( distn<rad, dtype=int)/(1+np.pi*(rad**2))
    
    
    return z3n, z4n, z5n, z6n, gret, gretvals
    
        