import numpy as np
import nibabel as nib
import torch
import math
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

def create_dictionary(image_dir,label_dir):
    lst = []
    for f in os.listdir(image_dir):
        if not f.startswith('.'):
            lst.append(f)
        else:
            pass
    lst.sort()
    label_lst=[]
    for f in os.listdir(label_dir):
        if not f.startswith('.'):
            label_lst.append(f)
        else:
            pass
    label_lst.sort()
   
    dir_dictionary={}
    for i in range(len(lst)):
        dir_dictionary[lst[i]]=label_lst[i]
        
    return dir_dictionary



def load_data_nii(fname):
    import nibabel as nib
    img = nib.load(fname)
    affine_mat=img.affine
    hdr=img.header
    data = img.get_fdata()
    data_norm = data
    return data_norm 

def calculate_l1_distance(img,img2):
    dist = np.sum(abs(img[:] - img2[:]));
    return dist

def calculate_l2_distance(img,img2):
    dist = np.sqrt(np.sum((img[:] - img2[:])** 2));
    return dist

def calculate_RMSE(img,img2):
    m=img.shape[0]
    n= img.shape[1]
    rmse = np.sqrt(np.sum((img[:] - img2[:])** 2)/(m*n));
    return rmse

def calculate_PSNR(img,img2):
    psnr = 10* math.log10( (np.sum(img[:]** 2)) / (np.sum((img[:] - img2[:])** 2)) )
    return psnr


def fourier_transform(image,shift=False):
    FT = np.fft.fft2(image)
 
    if shift:
        f_shift = np.fft.fftshift(FT)
        return f_shift
    else:
        return FT
    
def inverse_fourier_transform(data,shift=False):
    if shift:
        image = np.fft.ifft2(np.fft.ifftshift(data))
    else:
        image = np.fft.ifft2(data)
    return image

def save_img(img,name,fol_dir):
    figure(figsize=(8, 6), dpi=80)
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    plt.savefig(fol_dir+name+'.png', bbox_inches = 'tight',facecolor='white',pad_inches = 0) 
    plt.show()

def save_img_using_pil_lib(img,name,fol_dir,upsample=False, shape=None):
    data= img
    data = data.astype('float')
    data = (data/data.max())*255
    data = data.astype(np.uint8)
    data = Image.fromarray(data)
    if upsample:
        data = data.resize(shape,Image.BICUBIC)
    data.save(fol_dir+name+'.png')
    
def crop_center(arr,factor):
    y,x = arr.shape
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    return arr[starty:starty+(y//factor),startx:startx+(x//factor)]  

def pad_zeros_around(arr,factor,original_arr):
    y,x = original_arr.shape
    rows_pad =(y-(y//factor))//2
    cols_pad =(x-(x//factor))//2
    return np.pad(arr, [(rows_pad, rows_pad), (cols_pad, cols_pad)], mode='constant',constant_values=0)

def crop_pad_image_kspace(data,pad=False,factor=2):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    data = crop_center(arr=fshift,factor=factor)
    
    if pad:
        data= pad_zeros_around(data,factor,fshift)
    img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data))
    return np.abs(img_reco_cropped )

def crop_pad_kspace(data,pad=False,factor=2):  #function for cropping and/or padding the image in kspace
    F = np.fft.fft2(data)
    fshift = np.fft.fftshift(F)
    
    y,x = fshift.shape
    data_pad = np.zeros((y,x),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    
    arr = fshift[starty:starty+(y//factor),startx:startx+(x//factor)]
    if pad:
        data_pad[starty:starty+(y//factor),startx:startx+(x//factor)] = arr
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifft2(np.fft.ifftshift(arr)) 
    return np.abs(img_reco_cropped )



def normalize_image(image):
    max_img = image.max()
    min_img = image.min()
    denom = max_img-min_img
    norm_image = (image-min_img)/denom
    return norm_image

def get_gauss_filter(image,R=30,high=False):
    X = [i for i in range(image.shape[1])]
    Y = [i for i in range(image.shape[0])]
    
    Cy, Cx = image.shape
    val = 0.5

    X,Y = np.meshgrid(X, Y)
    low_filter = np.exp(-((X-(Cx*val))**2+(Y-(Cy*val))**2)/(2*R)**2)
    if high:
        return (1-low_filter)
    else:
        return low_filter


# ***********************************************************************************************************************************************************

def get_gaussian_filter(data,radius=20,low=True):
    #defining a low pass filter

    X = [i for i in range(data.shape[1])]
    Y = [i for i in range(data.shape[0])]

    Cy, Cx = data.shape
    # R= 30 and threshold= 0.001 downsampling_factor=2
    # R= 20 and threshold= 0.001 downsampling_factor=5
    # R= 17 and threshold= 0.001 downsampling_factor=7
    # R= 16 and threshold= 0.001 downsampling_factor=8
    val = 0.5  # if small then hr is same as image

    X,Y = np.meshgrid(X, Y)
    low_filter = np.exp(-((X-(Cx*val))**2+(Y-(Cy*val))**2)/(2*radius)**2)
    
    if low:
        return low_filter
    else:
        return 1-low_filter

    
def apply_filter(image_arr,filter_apply):
    F = np.fft.fft2(image_arr)  #fourier transform of image
    fshift = np.fft.fftshift(F)  #shifting 

    FFL = filter_apply* fshift  #multiplying with gaussian filter
    img_recon = np.abs(np.fft.ifft2(np.fft.ifftshift(FFL))) #inverse shift and inverse fourier transform
    
    return img_recon
    
    
def downsample_gaussian(image_arr,factor=2):
    if factor==2:
        radius = 25
    elif factor==4:
        radius=15
    elif factor==6:
        radius=11
    elif factor==8:
        radius =9
    else:
        print(f'downsample factor{factor} not implemented.Pass radius arg value')
    low_filter = get_gaussian_filter(image_arr,radius=radius)
    image_downsampled = apply_filter(image_arr,low_filter)
    return image_downsampled

# *************************************************************************************************************************************************


def downsample_bicubic(image_arr, factor=2):
    h, w = image_arr.shape
    new_height = int(h / factor)
    new_width = int(w / factor)

    # resize the image - down
    image_arr = cv2.resize(image_arr, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

    # resize the image - up
    image_arr = cv2.resize(image_arr, (w, h), interpolation = cv2.INTER_LINEAR)

    return image_arr


def save_image_cv(img,name,fol_dir):
    return (cv2.imwrite(fol_dir+name+'.png', img))