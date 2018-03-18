# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 18:33:08 2017

@author: TomoPC
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import sys
import h5py
from multiprocessing import Pool
#import multiprocessing as mp
import os
from DynamicSampling_TrainingClass import dynamic_image_training
#from DynamicSampling_ValidationClass import dynamic_image_validation
from DynamicSampling_ValidationClass_Patch import dynamic_image_validation
import pickle
from sklearn import linear_model
from scipy.misc import imsave
import os
import itertools
from functools import partial
from contextlib import contextmanager
import argparse


def training_iteration(c, filename, folder):

    RD=[]
    patches=[]
    features=[]
    sampling_percentage=[5,10,20,40,80]
    try:
        PrismImage=dynamic_image_training(folder+filename)

    except:
        print("File not found.")
        print(folder+filename)
      
    num_training_recon=5
    
    for ll in range(num_training_recon):
        for sampling in sampling_percentage:
            
            PrismImage.undersample(sampling)
            PrismImage.c=c
    #        print("Undersampling done \n")
            PrismImage.restore()
     #       print("Data restoration done \n")
            PrismImage.calculate_reduction_in_distortion()
      #      print("Reduction in Distortion done \n")
            PrismImage.computeFeatures()
       #     print("Calculated Features \n")
            PrismImage.patchify()
        #    print("Created Patches \n")
            
            if sampling==sampling_percentage[0]:
                RD=PrismImage.RD
                patches=PrismImage.patches[PrismImage.OrderForRD]
                features=PrismImage.features[PrismImage.OrderForRD]
            else:
            
                RD=np.append(RD, PrismImage.RD, axis=0)
                
                patches=np.append(patches, PrismImage.patches[PrismImage.OrderForRD], axis=0)
                
                features=np.append(features, PrismImage.features[PrismImage.OrderForRD], axis=0)
                
        print("Training Completed for c = {}." .format(c)) 
        folderpath=folder+"Training/"+filename.split('.')[0]+'/'
        if not os.path.exists(folderpath+str(c)+"c/"):
        
            #print(RD.shape)
            #print(patches.shape)
            #print(features.shape)
            os.makedirs(folderpath+str(c)+"c/")
            h5f = h5py.File(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".hdf5", "w")
            h5f.create_dataset('RD', data=RD,  maxshape=(None,), chunks=True)
            h5f.create_dataset('patches', data=patches,  maxshape=(None, 31,31), chunks=True)
            h5f.create_dataset('features', data=features,  maxshape=(None, 28), chunks=True)
            h5f.close()  
                
        else:
            print("Folder Already Exists, appending new data.")
            with h5py.File(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".hdf5", 'a') as hf:
                hf["RD"].resize((hf["RD"].shape[0] + RD.shape[0]), axis = 0)
                hf["RD"][-RD.shape[0]:] = RD
            
                hf["patches"].resize((hf["patches"].shape[0] + patches.shape[0]), axis = 0)
                hf["patches"][-patches.shape[0]:] = patches
            
                hf["features"].resize((hf["features"].shape[0] + features.shape[0]), axis = 0)
                hf["features"][-features.shape[0]:] = features
                


    pickle_out = open(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".pickle","wb")
    pickle.dump(PrismImage.return_params(), pickle_out)
    pickle_out.close()


def validation_iteration(c, filename, folder):

    stop_condition=15000
    
    reg = linear_model.LinearRegression()
    reg.fit(np.array([64,128,256,512]).reshape(-1,1), np.array([50,30,20,10]).reshape(-1,1))
    #(64x64):50, (128x128):30, (256x256):20, (512x512):10
    
    #filename='crazy.png'
    folderpath=folder +"Training/"+filename.split('.')[0]+'/'

                                    
    with open(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".pickle", 'rb') as handle:
        params = pickle.load(handle)
        
    handle.close()
    
    #let's choose the right parameters for inpainting first
        
    p_test=dynamic_image_validation(folder+filename, **(params))
    
    sampling_percentage=[5,10,20,40,80]
    
    p_vec=[2]
    
    p_error=np.zeros([len(sampling_percentage), len(p_vec)])
    
    #calculate best inpainting parameters

    i,j=0,0
    for sampling in sampling_percentage:
        
        p_test.undersample(sampling)
        j=0
        for p in p_vec:
            
            p_test.p=p
            p_test.restore()
            p_test.calculate_difference()
            
            p_error[i, j]=np.sum(p_test.difference)
            
            j+=1
        i+=1 
            
    p_error_index=np.trapz(p_error, axis=0)       
            
    best_p=p_vec[np.argmin(p_error_index)]
    
    test=dynamic_image_validation(folder+filename, **(params))
                  
    h5f = h5py.File(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".hdf5", "r") 
                                     
    test.RD=h5f['RD'][:]
    test.features=h5f['features'][:]
    h5f.close() 
    test.calculate_theta()
    test.p=best_p
    
    test.undersample(1)
    test.restore()
    test.computeFeaturesFull()
    test.predict()
    test.update_parameters()
    
    stop_condition=int((reg.predict(test.image_size[0])/100)*(test.image_size[0]*test.image_size[1]))
    
    print("Stopping condition has been calculated as {}" .format(stop_condition))
    for i in range(stop_condition):
            
        
        #print(str(i) +"\n")
        test.update_predictions_windowed()
        #print("Restored \n")
        test.update_parameters()
        
    test.restore()
    print("Completed_Validation for c = {}" .format(c))
    
#    
    start_condition=int(test.image_size[0]*test.image_size[1]*0.01)
#    
    percent_recon=np.linspace(start_condition,stop_condition,stop_condition/50).astype(int)
#    
    Distortion=[]
    
    for p in percent_recon:
        
        Dummy_validation=dynamic_image_validation(folder+filename, **(params))
        Dummy_validation.measured_values=test.measured_values[:p]
        Dummy_validation.unsampled_indices=test.unsampled_indices[:p]
        Dummy_validation.sampled_indices=test.sampled_indices[:p]
        Dummy_validation.measured_values=test.measured_values[:p]
        
        Dummy_validation.restore()
        Dummy_validation.calculate_difference()
        Distortion.append(np.sum(Dummy_validation.difference))
        
    TD=np.trapz(Distortion, x=percent_recon)
    
    np.save(folderpath+str(c)+'c/TD.npy', TD)
    np.save(folderpath+str(c)+'c/best_p.npy', best_p)
    print("Done with validation for c : {} ." .format(c))
    
    

def simulate(c, p, filename, folder):

   
    folderpath=folder+"Training/"+filename.split('.')[0]+'/'
    
                                    
    with open(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".pickle", 'rb') as handle:
        params = pickle.load(handle)
        
    handle.close()
        
    test=dynamic_image_validation(folder+filename, **(params))
                  
    h5f = h5py.File(folderpath+str(c)+'c/'+filename.split('.')[0]+"_"+str(c)+".hdf5", "r") 
                                     
    test.RD=h5f['RD'][:]
    test.features=h5f['features'][:]
    h5f.close() 
    test.calculate_theta()
    test.p=p
    #test.c=0.5
    
    test.undersample(1)
    test.restore()
    test.computeFeaturesFull()
    test.predict()
    test.update_parameters()
    
    stop_condition=int(test.image_size[0]*test.image_size[1]*0.1)
    
    for i in range(stop_condition):
            
        
        #print(str(i) +"\n")
        test.update_predictions_windowed()
        #print("Restored \n")
        test.update_parameters()
        
    test.restore()
    print("Completed_Validation for c = {}" .format(c))
    
    imsave(folderpath+"Mask.png", test.mask*255)
    imsave(folderpath+"Restored.png", test.restored_data)
    
    return test
    

#
#@contextmanager
#def poolcontext(*args, **kwargs):
#    pool = mp.Pool(*args, **kwargs)
#    yield pool
#    pool.terminate()        

    
if __name__ == '__main__':       
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--name", required=True,
    	help="path to the image")
    args = vars(ap.parse_args())
    
    filename=args["name"]

    folder=os.getcwd() +'\\'
    
                    
    folderpath=folder+'Training/'+filename.split('.')[0]+'/'
        
    sampling_percentage=[5,10,20,40,80]
    c_vec=[2,4,8,16,32]
    
    c_iterator = iter(c_vec)
    
#    with poolcontext(processes=(os.cpu_count()-2)) as pool:
#        pool.map(partial(training_iteration, filename=filename), c_iterator)
#        
    
    pool = Pool(os.cpu_count()-2)
    
    pool.map(partial(training_iteration, filename=filename, folder=folder),c_iterator) 
    pool.close()
    pool.join()
    
    c_iterator=iter(c_vec)
    
#    with poolcontext(processes=(os.cpu_count()-2)) as pool:
#        pool.map(partial(validation_iteration, filename=filename), c_iterator)
#        
    print("Calculating Inpainting Parameters:")
    pool=Pool(os.cpu_count()-2)
    pool.map(partial(validation_iteration, filename=filename, folder=folder), c_iterator)
    pool.close()
    pool.join()
    
    l=[]    
    for c in c_vec:
    
        l.append(np.load(folderpath+str(c)+'c/TD.npy'))

    tmp_index=np.argmin(l)
    
    best_p=np.load(folderpath +str(c_vec[tmp_index]) + 'c/best_p.npy')
    
    print("Best c was found to be: {}" .format(c_vec[tmp_index]))
    print("Best p was found to be: {}" .format(best_p))
    sampled=simulate(c_vec[tmp_index], best_p, filename, folder)


