import h5py
import numpy as np
import pydicom
import os
import glob
from sklearn.feature_extraction import image
from skimage.util import view_as_windows
import scipy.io as sio


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels

def write_hdf5(data,labels, output_filename):
    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)
        h.create_dataset('label', data=labels, shape=labels.shape)


def load_pixel_array(dcmdir):
    files_with_dcm = glob.glob(dcmdir+'\*.IMA')
    num_images = len(files_with_dcm)
    x = np.zeros((num_images,512,512))
    for i,path in enumerate(files_with_dcm):
        ds = pydicom.dcmread(path)
        x[i,] = ds.pixel_array.astype(np.float32)   
        
    return x


def make_train_individual(basedir,subdirs,xdir,ydir,patch_size,patch_per_image):
#    basedir = r'C:\Users\s2ataei\Documents\Dataset'
#    subdirs = ['\L067','\L096','\L109','\L143','\L192','\L286','\L291','\L310','\L333','\L506']
    ydir = '\\full_1mm'
    xdir = '\\quarter_1mm'
    for folder in subdirs:
        print(basedir+folder+xdir)
        x,y = extract_patches(basedir+folder+xdir,basedir+folder+ydir,(patch_size),patch_per_image)
        savedir = basedir+folder + '\\train_patches_exclude_air_64.h5'
        print(savedir)
        write_hdf5(x,y,savedir)
        
    return 

def make_train_combined(basedir,subdirs,individual_name):
#    basedir = r'C:\Users\s2ataei\Documents\Dataset'
#    subdirs = ['\L067','\L096','\L143','\L192','\L286','\L291','\L310','\L333']
#    individual_name = '\\train_patches_exclude_air_64'
    Low = [None] * len(subdirs)
    High = [None] * len(subdirs)
    for i,folder in enumerate(subdirs):
        Low[i],High[i] = read_hdf5(basedir+folder+individual_name)
        
    return np.vstack(Low).astype('float32'),np.vstack(High).astype('float32')
        
       
def exclude_air(x_patches,y_patches,thresh):
    num_patches = x_patches.shape[0]
    indicies = []
    for i in range (num_patches):
        if ((np.mean(x_patches[i,:,:])) < thresh):
            indicies.append(i)
                      
    x_r = np.delete(x_patches,indicies,0)
    y_r = np.delete(y_patches,indicies,0)       
    
    return x_r,y_r  
        
            
def exclude_from_set(basedir,subdirs,thresh):
    for folder in subdirs:
        x,y = read_hdf5(basedir+folder+'\\train_patches_64.h5')
        x,y = exclude_air(x,y,thresh)
        write_hdf5(x,y,basedir+folder+'\\train_patches_exclude_air_64')
        
    return


def extract_patches(im, window_shape=(64,64),stride=64):
    num_imgs = im.shape[0]
    p = [None] * num_imgs
    for i in range(num_imgs):
        p[i] = np.vstack(view_as_windows(im[i,:,:], window_shape, stride))
        
    return np.vstack(p)


def remove_air():
    basedir = r'C:\Users\s2ataei\Documents\Dataset'
    subdirs = ['\L067','\L096','\L143','\L192','\L286','\L291']

    for folder in subdirs:  
        full = sio.loadmat(basedir+folder+'\\full_1mm\contrast_adjusted.mat')['fslicesg']
        quarter = sio.loadmat(basedir+folder+'\\quarter_1mm\contrast_adjusted.mat')['qslicesg']
        full = extract_patches(full)
        quarter = extract_patches(quarter)
        c = 0
        for i in range(quarter.shape[0]):
            if (np.max(quarter[i,:,:])!=0):
                c+=1
        nf = np.zeros([c,64,64])
        nq = np.zeros([c,64,64])
        c=0
        for i in range(quarter.shape[0]):
            if (np.max(quarter[i,:,:])!=0):
                nq[c,:,:] = quarter[i,:,:]
                nf[c,:,:] = full[i,:,:]
                c+=1
        write_hdf5(nq,nf,basedir+folder+'\\tpatches.h5')
        
        return