import utility as ut
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-image-dir', type=str)
    args = parser.parse_args()

    # # Z-axis Images
    # ### F1 160

    datapath_hr = '../OneDrive_1_4-8-2022/AD_P522R_F1_160/MRI_25um/mag_sos_wn.nii'
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../dataset/z_axis/'
    label_train_dir = '../dataset/z_axis/label/train/'
    label_val_dir ='../dataset/z_axis/label/val/'
    hr_path = 'hr_f1_160'

    factor = [2,4,6,8]

    if not os.path.exists(label_train_dir):
       os.makedirs(label_train_dir)
    if not os.path.exists(label_val_dir):
       os.makedirs(label_val_dir)
    for fac in factor:
        train_path = main_path+ "factor_{}/train/".format(fac)
        val_path = main_path+ "factor_{}/val/".format(fac)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        


    for fac in factor:
        x,y,z = data_hr.shape
        k=42
        for i in range(71):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f1_160_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
    
    # saving lr images
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# ******************************************************************************************************************************

    # ## F2 145
    datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F2_145/MRI_25um/mag_sos_wn.nii'
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../dataset/z_axis/'
    hr_path ='hr_f2_145'

    factor = [2,4,6,8]

    for fac in factor:
        x,y,z = data_hr.shape
        k=42
        for i in range(71):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f2_145_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
     
    # saving lr images
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# *********************************************************************************************************************************

# ## F4 149

# ## Validation set

    datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F4_149/MRI_25um/mag_sos_wn.nii'
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../dataset/z_axis/'
    hr_path = 'hr_f4_149'

    factor = [2,4,6,8]

    for fac in factor:
        x,y,z = data_hr.shape
        k=42
        for i in range(71):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f4_149_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/val/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)

    
    # saving lr images
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_val_dir)



# ***********************************************************************************************************************
# ## F5 153

    datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii'
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../dataset/z_axis/'
    sub_path_2 ='lr_f5_153_2'
    sub_path_4='lr_f5_153_4'
    sub_path_8 ='lr_f5_153_8'
    hr_path ='hr_f5_153'

    factor = [2,4,6,8]

    for fac in factor:
        x,y,z = data_hr.shape
        k=42
        for i in range(71):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f5_153_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
   
    # saving lr images
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)


