# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:32:22 2018

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
from sklearn import linear_model


def computePolyFeatures(Feature):
    
    PolyFeatures = np.hstack([np.ones((np.shape(Feature)[0],1)),Feature])
    for i in range(0,np.shape(Feature)[1]):
        for j in range(i,np.shape(Feature)[1]):
            PolyFeatures = np.column_stack([PolyFeatures,Feature[:,i]*Feature[:,j]])

    return PolyFeatures



class dynamic_image_validation:
    
    def __init__(self, filename, **kwargs):
        
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

        for key in kwargs:
            setattr(self, key, kwargs[key])
#        
        self.data=self.data.astype(np.uint16)
        self.p=2
        self.NumNbrs=10
        self.WindowSize=15
        self.patch_size=self.WindowSize*2
        self.PercOfRD=20
        self.c=2
        self.image_size=np.shape(img)
        self.featdistcutoff=0.25
        self.filename=filename
        self.MinRadius=3
        self.MaxRadius=10     
        self.FeatDistCutoff = 0.25
        self.MaxWindowForTraining=15
        
        for key in kwargs:
                setattr(self, key, kwargs[key])
                        
        
        
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
        
#    def restore(self):
#        
#        self.restored_data=ImRestore(self.sampled_indices, self.unsampled_indices, self.measured_values, self.image_size, self.c).astype(np.uint16)                
#        #Navier Stokes is more accurate, but Telea is faster        
#        
##        self.restored_data = cv2.inpaint(self.undersampled_image,
##                                  np.abs((self.mask-1)),
##                                  3,cv2.INPAINT_NS)
##        self.restored_data_0 = cv2.inpaint(self.undersampled_image,
##                                  np.abs((self.mask-1)),
##                                  3,cv2.INPAINT_TELEA)
#        self.calculate_difference()
        
    def calculate_difference(self):
        if self.dynamic_type == 'D':
            difference=self.data!=self.restored_data
            self.difference =difference.astype('float')
        if self.dynamic_type == 'C':
            self.difference=abs((self.restored_data.astype(np.float32))-(self.data.astype(np.float32)))


    def find_neighbors(self, patch='N'):
        
        
        
        if patch=='N':
            
            Neigh = NearestNeighbors(n_neighbors=self.NumNbrs)
            Neigh.fit(self.sampled_indices)
            self.NeighborDistances, self.NeighborIndices = Neigh.kneighbors(self.unsampled_indices)
            self.NeighborValues=self.measured_values[self.NeighborIndices]
            
        if patch=='Y':
            
            Neigh = NearestNeighbors(n_neighbors=self.NumNbrs)
            Neigh.fit(self.sampled_indices)
            self.NeighborDistances, self.NeighborIndices = Neigh.kneighbors(self.small_unsampled_indices)
            self.NeighborValues=self.measured_values[self.NeighborIndices]
             
            

    def computeNeighborWeights(self):
        
        UnNormNeighborWeights=1/np.power(self.NeighborDistances,self.p)
        SumOverRow = (np.sum(UnNormNeighborWeights,axis=1))
        self.NeighborWeights=UnNormNeighborWeights/SumOverRow[:, np.newaxis]

        
        
    def computeFeaturesFull(self):
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
        NeighborValues=self.measured_values[self.NeighborIndices]
        
        if self.dynamic_type == 'D':
            DiffVect=NeighborValues!=np.transpose(np.matlib.repmat(ReconValues,np.shape(NeighborValues)[1],1))
            DiffVect=DiffVect.astype('float')
        if self.dynamic_type == 'C':
            DiffVect=abs(NeighborValues-np.transpose(np.matlib.repmat(ReconValues,np.shape(NeighborValues)[1],1)))
        Feature_0 = np.sum(self.NeighborWeights*DiffVect,axis=1)
        Feature_1 = np.sqrt((1/self.NumNbrs)*np.sum(np.power(DiffVect,2),axis=1))
        return(Feature_0,Feature_1)
    
    
    def update_parameters(self):
        
        #Everythin gets updated        
    
        tmp_index=np.argmax(self.predictions)
        
        new_measurement=np.reshape(self.unsampled_indices[tmp_index], [1,2])
        new_value=self.data[new_measurement[:,0], new_measurement[:,1]]
        
        
        
        self.sampled_indices = np.concatenate([self.sampled_indices, new_measurement])
        self.measured_values = np.concatenate([self.measured_values, new_value])
        
        self.unsampled_indices=np.delete(self.unsampled_indices, tmp_index, axis=0)
        self.predictions=np.delete(self.predictions, tmp_index, axis=0)        
        self.mask[new_measurement[:,0], new_measurement[:,1]]=1
        self.undersampled_image[new_measurement[:,0], new_measurement[:,1]]=new_value
                               
    def calculate_theta(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.features, self.RD)
        theta = np.zeros((self.features.shape[1]))    
        if self.dynamic_type=='D':            
            theta[0:24]=regr.coef_[0:24]
            theta[26]=regr.coef_[25]
        else:
            theta = regr.coef_ 
            
        self.theta=theta
        
        

    def predict(self):
    
        self.predictions=np.dot(self.features, self.theta)
        
    
    def update_predictions_windowed(self):
        
        NumSamples=len(self.measured_values)
        SuggestedRadius = int(np.sqrt((1/np.pi)*(self.image_size[0]*self.image_size[1]*self.NumNbrs/NumSamples)))
        UpdateRadiusTemp=np.max([SuggestedRadius,self.MinRadius]);
        UpdateRadius=int(np.min([self.MaxRadius,UpdateRadiusTemp]));                       
        updateRadiusMat = np.zeros((self.image_size[0],self.image_size[1]))
        Done=0
        while(Done==0):
            updateRadiusMat[max(self.sampled_indices[-1][0]-UpdateRadius,0):min(self.sampled_indices[-1][0]+UpdateRadius,self.image_size[0])][:,max(self.sampled_indices[-1][1]-UpdateRadius,0):min(self.sampled_indices[-1][1]+UpdateRadius,self.image_size[1])]=1

            updateIdxs = np.where(updateRadiusMat[self.mask==0]==1)
            
            SmallUnMeasuredIdxs = np.transpose(np.where(np.logical_and(self.mask==0,updateRadiusMat==1)))
            if SmallUnMeasuredIdxs.size==0:
                UpdateRadius=int(UpdateRadius*1.5)
            else:
                Done=1
                
        self.small_unsampled_indices=SmallUnMeasuredIdxs

        self.find_neighbors(patch='Y')        
        
        self.computeNeighborWeights()
        
        self.SmallReconValues=self.restore_window()

        self.restored_data[(np.logical_and(self.mask==0,updateRadiusMat==1))]=self.SmallReconValues
        #ReconImage[MeasuredIdxs[:,0],MeasuredIdxs[:,1]]=MeasuredValues
    
        # Compute features
        self.computeFeaturesWindow()
        # Compute ERD
        SmallERDValues = np.dot(self.features, self.theta)
    
        #self.restored_data[updateIdxs] = SmallReconValues
        self.predictions[updateIdxs] = SmallERDValues
               
    def restore_window(self):
        
        #self.find_neighbors()
        #self.computeNeighborWeights()
        if self.dynamic_type=='D':
            
            ClassLabels = np.unique(self.NeighborValues)
            ClassWeightSums = np.zeros((np.shape(self.NeighborWeights)[0],np.shape(ClassLabels)[0]))
            for i in range(0,np.shape(ClassLabels)[0]):
                TempFeats=np.zeros((np.shape(self.NeighborWeights)[0],np.shape(self.NeighborWeights)[1]))
                np.copyto(TempFeats,self.NeighborWeights)
                TempFeats[self.NeighborValues!=ClassLabels[i]]=0
                ClassWeightSums[:,i]=np.sum(TempFeats,axis=1)
            IdxOfMaxClass = np.argmax(ClassWeightSums,axis=1)
            ReconValues = ClassLabels[IdxOfMaxClass]
            
        if self.dynamic_type == 'C':
            ReconValues=np.sum(self.NeighborValues*self.NeighborWeights,axis=1)
            
        return ReconValues.astype(np.int16)
           
     
    def computeFeaturesWindow(self):

        Feature=np.zeros((np.shape(self.small_unsampled_indices)[0],6))
        
        
        # Compute st div features
        Feature[:,0],Feature[:,1]=self.computeStDivFeaturesWindow()
        
        # Compute distance/density features
        Feature[:,2],Feature[:,3]=self.computeDensityDistanceFeatures()
        
        GradientImageX,GradientImageY=self.computeGradientFeatures()
        Feature[:,4] = GradientImageY[self.small_unsampled_indices[:,0],self.small_unsampled_indices[:,1]]
        Feature[:,5] = GradientImageX[self.small_unsampled_indices[:,0],self.small_unsampled_indices[:,1]]
        PolyFeatures = computePolyFeatures(Feature)
        
        self.features=PolyFeatures
        
        

    def computeStDivFeaturesWindow(self):
    
        
        if self.dynamic_type == 'D':
            DiffVect=self.NeighborValues!=np.transpose(np.matlib.repmat(self.SmallReconValues,np.shape(self.NeighborValues)[1],1))
            DiffVect=DiffVect.astype('float')
        if self.dynamic_type == 'C':
            DiffVect=abs(self.NeighborValues-np.transpose(np.matlib.repmat(self.SmallReconValues,np.shape(self.NeighborValues)[1],1)))
        Feature_0 = np.sum(self.NeighborWeights*DiffVect,axis=1)
        Feature_1 = np.sqrt((1/self.NumNbrs)*np.sum(np.power(DiffVect,2),axis=1))
        return(Feature_0,Feature_1)       


    

