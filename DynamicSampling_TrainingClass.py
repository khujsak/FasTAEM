# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:26:21 2018

@author: TomoPC
"""

from scipy import misc
import numpy as np
from skimage import color
import matplotlib.pyplot as plt
import random
import math
#import cv2
from sklearn.neighbors import NearestNeighbors
import sys
import scipy
#import cv2
import h5py
from multiprocessing import Pool
import os

class dynamic_image_training:
    
    def __init__(self, filename):
        
        #The initial class creation routine
        #Checks if a file is Rgb or gray, and does conversion
        #For now automatically converts data to uint8, but does
        #A simple check first to tell if data is continuous or discrete
        
        img=misc.imread(filename)
        if img.ndim >2:
            self.data=color.rgb2gray(img)*256

        self.dynamic_type='C'   
#        if 'uint8' in str(self.data):
#            self.dynamic_type = 'D'
#        else:
#            self.dynamic_type = 'C'
        
        self.data=self.data.astype(np.uint16)
        self.p=2
        self.NumNbrs=10
        self.WindowSize=15
        self.patch_size=self.WindowSize*2
        self.PercOfRD=20
        self.gauss_kern=gauss_kern
        self.c=2
        self.image_size=np.shape(img)
        self.featdistcutoff=0.25
        
    def return_params(self,):
        
        params={"c":self.c, "p":self.p, "NumNbrs":self.NumNbrs, \
                "WindowSize":self.WindowSize, "patch_size":self.patch_size, \
                "featdistcutoff":0.25}
        
        return params
        
    def get_type(self, ):
        #Helper Function to get data type
         return self.data.dtype
     
    def patchify(self, edge_handling='clip'):
        #Extracting patches in fast vectorized form from indices
    
        self.edge_condition=edge_handling
        m,n = self.data.shape
        try:
            patch_size=self.patch_size
        except:
            patch_size = input("Enter patch size: ")
            
        indices_row=self.unsampled_indices[:,0]
        indices_column=self.unsampled_indices[:,1]
        
        K = int(np.floor(patch_size/2.0))
        R = np.arange(-K,K+1)                  
        self.patches = np.take(self.data,R[:,None]*n + R + (indices_row*n+indices_column)[:,None,None], mode='clip')
        
    def undersample(self, sampling_percentage=None):
        #Choose a random set of column and row values to set to zero
        #Recieves the random percent or a flag to be random
        random.seed()
        m,n = self.data.shape 
        mu=0
        sigma=1
        mask=np.zeros(m*n).astype(np.uint8)
        
        if sampling_percentage is None:
            number_sampled_points=np.abs((random.gauss(mu, sigma)))
            number_sampled_points*=m*n*0.2  
            number_sampled_points=int(number_sampled_points)
        else:
            number_sampled_points=int(m*n*sampling_percentage/100)
            
        sampled_points=np.array(random.sample(range(m*n-1),number_sampled_points))
        print("Sampling Percentage for this round: {}" .format(len(sampled_points)/(m*n)))

        mask[sampled_points]=1
        mask=np.reshape(mask, [m,n])
        
        self.mask=mask
        self.undersampled_image=mask*self.data
        self.sample_percentage=number_sampled_points/(m*n)
        self.sampled_indices=np.array(np.unravel_index(sampled_points, [m,n])).T
        self.unsampled_indices=np.array(np.where(self.mask==0)).T
        self.measured_values= self.data[self.sampled_indices[:,0], self.sampled_indices[:,1]]
        
    def restore(self):
        
        self.find_neighbors()
        self.computeNeighborWeights()
        if self.dynamic_type=='D':
            
            ClassLabels = np.unique(self.NeighborValues)
            ClassWeightSums = np.zeros((np.shape(self.NeighborWeights)[0],np.shape(self.ClassLabels)[0]))
            for i in range(0,np.shape(self.ClassLabels)[0]):
                TempFeats=np.zeros((np.shape(self.NeighborWeights)[0],np.shape(self.NeighborWeights)[1]))
                np.copyto(TempFeats,self.NeighborWeights)
                TempFeats[self.NeighborValues!=ClassLabels[i]]=0
                ClassWeightSums[:,i]=np.sum(TempFeats,axis=1)
            IdxOfMaxClass = np.argmax(ClassWeightSums,axis=1)
            ReconValues = ClassLabels[IdxOfMaxClass]
            
        if self.dynamic_type == 'C':
            ReconValues=np.sum(self.NeighborValues*self.NeighborWeights,axis=1)
        
        ReconImage = np.zeros((self.image_size[0],self.image_size[1]))
        ReconImage[self.unsampled_indices[:,0],self.unsampled_indices[:,1]]=ReconValues
        ReconImage[self.sampled_indices[:,0],self.sampled_indices[:,1]]=self.measured_values
        
        self.restored_data=ReconImage.astype(np.int16)
        self.calculate_difference()
        
        
    def calculate_difference(self):
        if self.dynamic_type == 'D':
            difference=self.data!=self.restored_data
            self.difference =difference.astype('float')
        if self.dynamic_type == 'C':
            self.difference=abs((self.restored_data.astype(np.float32))-(self.data.astype(np.float32)))

    def calculate_reduction_in_distortion(self):
        m,n = self.data.shape
        self.calculate_difference()
        difference=self.difference.astype('int')
        difference_padded = np.lib.pad(difference,(int(np.floor(self.WindowSize/2)),int(np.floor(self.WindowSize/2))),'constant',constant_values=0)
        WindowSize=int(self.WindowSize)
        NumRandChoices =  int(self.PercOfRD*self.unsampled_indices.shape[0]/100)
        OrderForRD = random.sample(range(0,self.unsampled_indices.shape[0]), NumRandChoices) 
        self.OrderForRD=OrderForRD

 #Want to take padded image and return list of patches at 
#Unsampled places with correct patches       
        
        padded_unsampled_indices=np.add(self.unsampled_indices, int(np.floor(self.WindowSize/2)))
        K = int(np.floor(WindowSize/2.0))
        R = np.arange(-K,K+1)   
        ImgAsBlocksOnlyUnmeasured=np.take(difference_padded,R[:,None]*n + R + (padded_unsampled_indices[:,0]*n+padded_unsampled_indices[:,1])[:,None,None], mode='raise')
        ImgAsBlocksOnlyUnmeasured=np.reshape(ImgAsBlocksOnlyUnmeasured, [WindowSize*WindowSize, self.unsampled_indices.shape[0]])
        
        #Is this right??
        self.patches=ImgAsBlocksOnlyUnmeasured[:,OrderForRD]
        temp = np.zeros((WindowSize*WindowSize,NumRandChoices))
        self.find_neighbors()
        sigma = self.NeighborDistances[:,0]/self.c
        cnt = 0;
        for l in OrderForRD:
            Filter = gauss_kern(sigma[l],WindowSize)
            temp[:,cnt] = ImgAsBlocksOnlyUnmeasured[:,l]*Filter
            cnt=cnt+1
        self.RD = np.sum(temp, axis=0)
        
    def find_neighbors(self):
        Neigh = NearestNeighbors(n_neighbors=self.NumNbrs)
        Neigh.fit(self.sampled_indices)
        self.NeighborDistances, self.NeighborIndices = Neigh.kneighbors(self.unsampled_indices)
        self.NeighborValues=self.measured_values[self.NeighborIndices]

    def computeNeighborWeights(self):
        
        UnNormNeighborWeights=1/np.power(self.NeighborDistances,self.p)
        SumOverRow = (np.sum(UnNormNeighborWeights,axis=1))
        self.NeighborWeights=UnNormNeighborWeights/SumOverRow[:, np.newaxis]
        
        
        
    def computeFeatures(self):
        #MeasuredValues,MeasuredIdxs,UnMeasuredIdxs,SizeImage,NeighborValues,NeighborWeights,NeighborDistances,TrainingInfo,ReconValues,ReconImage,Resolution,ImageType
        Feature=np.zeros((np.shape(self.unsampled_indices)[0],6))
        
        
        
        # Compute st div features
        Feature[:,0],Feature[:,1]=self.computeStDivFeatures()
        
        # Compute distance/density features
        Feature[:,2],Feature[:,3]=self.computeDensityDistanceFeatures()
        
        GradientImageX,GradientImageY=self.computeGradientFeatures()
        Feature[:,4] = GradientImageY[self.unsampled_indices[:,0],self.unsampled_indices[:,1]]
        Feature[:,5] = GradientImageX[self.unsampled_indices[:,0],self.unsampled_indices[:,1]]

        
        PolyFeatures = computePolyFeatures(Feature)
        
        self.features=PolyFeatures
        
        
    def computeGradientFeatures(self):
        GradientImageX,GradientImageY = np.gradient(self.restored_data)
        if self.dynamic_type=='D':
            GradientImageX[GradientImageX!=0]=1
            GradientImageY[GradientImageY!=0]=1
        elif self.dynamic_type=='C':
            GradientImageX=abs(GradientImageX)
            GradientImageY=abs(GradientImageY)
        return(GradientImageX,GradientImageY)
    #    
    def computeDensityDistanceFeatures(self):
    
        CutoffDist = np.ceil(np.sqrt((self.featdistcutoff/100)*(self.image_size[0]*self.image_size[1]/np.pi)))
        Feature_2 = self.NeighborDistances[:,0]
        NeighborsInCircle=np.sum(self.NeighborDistances<=CutoffDist,axis=1)
        Feature_3 = (1+(np.pi*(np.power(CutoffDist,2))))/(1+NeighborsInCircle)
        return(Feature_2,Feature_3)


    def computeStDivFeatures(self):
        
        ReconValues=self.restored_data[self.unsampled_indices[:,0], self.unsampled_indices[:,1]]
        #NeighborValues=self.measured_values[self.NeighborIndices]
        
        if self.dynamic_type == 'D':
            DiffVect=self.NeighborValues!=np.transpose(np.matlib.repmat(ReconValues,np.shape(self.NeighborValues)[1],1))
            DiffVect=DiffVect.astype('float')
        if self.dynamic_type == 'C':
            DiffVect=abs(self.NeighborValues-np.transpose(np.matlib.repmat(ReconValues,np.shape(self.NeighborValues)[1],1)))
        Feature_0 = np.sum(self.NeighborWeights*DiffVect,axis=1)
        Feature_1 = np.sqrt((1/self.NumNbrs)*np.sum(np.power(DiffVect,2),axis=1))
        return(Feature_0,Feature_1)
    
def computePolyFeatures(Feature):
    
    PolyFeatures = np.hstack([np.ones((np.shape(Feature)[0],1)),Feature])
    for i in range(0,np.shape(Feature)[1]):
        for j in range(i,np.shape(Feature)[1]):
            PolyFeatures = np.column_stack([PolyFeatures,Feature[:,i]*Feature[:,j]])

    return PolyFeatures

    
    #    
def gauss_kern(sigma, size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(np.floor(size/2))
    sizey = size
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-(x**2+y**2) / (2*(sigma)**2))
    return np.ravel(g / g.max())

