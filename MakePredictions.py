# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:12:37 2017

@author: tomosys
"""


import numpy as np
from skimage.measure import structural_similarity as ssim

from dynsamp import random_inpaint, mse, compare_images, distance_calc_pred, update_featurevector


#Predict


#Construct Image of Dimension Sc (Reshape)

sc=256

#Sample this many points randomly



points=1000


#Number of nearest measurements to keep in SD Value
L=6

#Need to define radius of z6 as a function of image area

rad=.2*sc


####Read in and Resize Image

os.chdir('/home/tomosys/Documents/SLADS/')
im = cv2.imread('100.bmp',0)

im=im[120:120+sc, 120:120+sc]

im=np.rint(im/44)


sz=np.shape(im)

imr=np.ravel(im)


ind=imr.size


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

#image_result = ptv.tv1_2d(image_result, 5)

imrr=np.ravel(image_result)

uspv=imrr[usp]


#
#plt.figure()
#plt.imshow(image_result, interpolation='none')
#plt.figure()
#plt.imshow(im, interpolation='none')
#plt.figure()
#plt.imshow(imsr, interpolation='none')



#initialize values for feature vecors


rr=np.size(usp)




updis=np.zeros(rr)
z3=np.zeros(rr)
z4=np.zeros(rr)

start = time.time()
print("DistanceCalc")

uspv=np.reshape(uspv, [sc*sc-points, 1])
updis, z3, z4, z6, gret, gretvals=distance_calc_pred(rr, usp, sp, spv, uspv, rad, L, disbig, sc, points, imr)

gx = np.gradient(image_result, axis=0)
gy = np.gradient(image_result, axis=1)

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
#Z=np.transpose(Z)


#Make prediction here once feature matrix is calculated


"""
FOR LOOP HERE
For Simulated Sampling
"""

for k in range(9000):

    
    Vest=np.matmul(np.reshape(coeff[0], [1,28]), Z)
    
    print('Estimated ERD: ' + str(np.max(Vest)))
    
    
    #Get index
       
    snew=np.argmax(Vest)
    imsnew=usp[0][snew]
    
    #delid=np.where(usp[0]==snew)
    
    #Need to update sp up, spv, upv with snew
    
    
    sp=np.append(sp, imsnew)
    spv=imr[sp]
    
    #indexus=np.where(usp==snew)
    usp=np.delete(usp, snew, axis=1)
    
    
    
    gret=np.delete(gret, snew, axis=0)
    gretvals=np.delete(gretvals, snew, axis=0)
    
    z6=np.delete(z6, snew)
    
    rr=np.size(usp)
    
    
    
    
    #Restore image with new point
    
    ims[sp]=imr[sp]
    
    
    #Inpaint Unsampled Image
    
    imsr=np.reshape(ims, sz)
    mask=np.ones(ind).astype(np.uint8)
    mask[sp]=0
    mask=np.reshape(mask, sz)
    
    #image_result = inpaint.inpaint_biharmonic(imsr, mask, multichannel=False)
    
    #dst = cv2.inpaint(imsr,dummat,6,cv2.INPAINT_TELEA)
    
    image_resultn = cv2.inpaint(imsr,mask,6,cv2.INPAINT_NS)
    
    #image_resultn = ptv.tv1_2d(image_resultn, 5)
    
    imrrn=np.ravel(image_resultn)

    uspv=imrrn[usp]
        
    
    #Update Features Vector
    
    z3, z4, z5, z6, gret, gretvals = update_featurevector(rr, usp, uspv, rad, L, disbig, gret, gretvals, z6, imr, imsnew)    
    
    
    
    
    
    gx = np.gradient(image_resultn, axis=0)
    gy = np.gradient(image_resultn, axis=1)
    
    gxr=np.abs(np.ravel(gx))
    gyr=np.abs(np.ravel(gy))
    
    z1=np.reshape(gxr[usp], [rr])
    z2=np.reshape(gyr[usp], [rr])
    
    
    
    Z=[z1, z2, z3, z4, z5, z6]
    
    for i in range(6):
        
        
        for j in range(6):
        
            
            if i>j:
                pass
            else:
                dummy=Z[i][:] * Z[j][:]
                Z.append(dummy)
    
    
    
    Z=np.array(Z)
    
    Z=np.append(np.ones([1,rr]), Z, 0)
    #Z=np.transpose(Z)
    
    points+=1
    print('iteration # ', k)


compare_images(image_resultn, im.astype(np.uint8), 'Dynamic Sampled')


spim=np.zeros([256*256])
spim[sp]=1
spim=np.reshape(spim, [256,256])

compare_images(spim, im, 'Dynamic Sampled Pattern')

randinpaint=random_inpaint(imr, 10000, sz)

compare_images(randinpaint, im.astype(np.uint8), 'Random Sampled')