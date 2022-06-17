
# write a shell command to create a directory and generate the patches

import numpy as np
from PIL import Image
import torch
import argparse

import  cv2, os

import nibabel as nib
import warnings
warnings.filterwarnings("ignore")


def create_index(db_size, cube_size, xmax, ymax, zmax):
    index_list = []
    x_index = torch.randint(0,xmax-cube_size, (db_size,1))
    y_index = torch.randint(0,ymax-cube_size, (db_size,1))
    z_index = torch.randint(0,zmax-cube_size, (db_size,1))
    for (x,y,z) in zip (x_index, y_index, z_index):
        index_list.append((x.item(),y.item(),z.item()))
    return index_list

def load_data_nii(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    data_norm = torch.from_numpy(data)
    return data_norm 

def preprocess_data(data,factor=2,pad=True):
    spectrum_3d = np.fft.fftn(data)  # Fourier transform along Y, X and T axes to obtain ky, kx, f
    spectrum_3d_sh = np.fft.fftshift(spectrum_3d, axes=(0,1,2))  # Apply frequency shift along spatial dimentions so

    x,y,z = spectrum_3d_sh.shape
    data_pad = np.zeros((x,y,z),dtype=np.complex_)
    center_y = y//2 #defining the center of image in x and y direction
    center_x = x//2
    center_z = z//2
    startx = center_x-(x//(factor*2))  
    starty = center_y-(y//(factor*2))
    startz = center_z-(z//(factor*2))
    arr = spectrum_3d_sh[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)]   
    if pad:
        data_pad[startx:startx+(x//factor),starty:starty+(y//factor),startz:startz+(z//factor)] = arr
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(data_pad))
    else:
        img_reco_cropped = np.fft.ifftn(np.fft.ifftshift(arr)) 
    return np.abs(img_reco_cropped)



def create_patch(image_arr, label_arr,patch_size,output_image_dir,output_label_dir,indexes,name):
    for i,idx in enumerate(indexes):
        crop_image,crop_label = extract_patch(image_arr,label_arr,idx,patch_size)
        img_name = name +'_input_p'+str(i)
        lbl_name = name +'_label_p'+str(i)
        save_patch_cube(crop_image,img_name,output_image_dir)
        save_patch_cube(crop_label,lbl_name,output_label_dir)
    return 1


def extract_patch(img_arr,label_arr,index,cube_size):
    x_index,y_index,z_index = index
    image = img_arr [x_index:x_index+cube_size,y_index:y_index+cube_size,z_index:z_index+cube_size]
    label = label_arr [x_index:x_index+cube_size,y_index:y_index+cube_size,z_index:z_index+cube_size]
       
    return image,label

def save_patch_cube(data,name,fol_dir):
    np.save(fol_dir+name+'.npy',data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)  #directory of nii file
    parser.add_argument('--factor', type=int, required=True)
    parser.add_argument('--name', type=str, required=True) #name whether f1_25 or f3_50 
    # parser.add_argument('--label-dir', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--num-patch', type=int, default=10)
    parser.add_argument('--output-image-dir', type=str, required=True)
    parser.add_argument('--output-label-dir', type=str, required=True)
    args = parser.parse_args()

    label = load_data_nii(args.data_dir)
    xmax,ymax,zmax = label.shape
    indexes = create_index(args.num_patch, args.patch_size, xmax, ymax, zmax)
    images = preprocess_data(label,factor=args.factor,pad=True)
    
    create_patch (images,label,args.patch_size,args.output_image_dir,args.output_label_dir,indexes,args.name)
