import utility as ut
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size',type=str,required=False,default='MRI_25um')
    args = parser.parse_args()

    factor = [2,4,6,8]

    dataset_size = args.dataset_size
    
    if dataset_size =='MRI_25um':
        dataset_name = 'bicubic_dataset25'
        start = 20
        end= 260
        num_img = end-start
    else:
       dataset_name = 'bicubic_dataset50' 
       start = 42
       end = 112
       num_img = end-start

    # # Z-axis Images
    # ### F1 160
    datapath_hr = '../OneDrive/AD_P522R_F1_160/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    main_path= '../{}/z_axis/'.format(dataset_name)
    label_train_dir = '../{}/z_axis/label/train/'.format(dataset_name)
    label_val_dir ='../{}/z_axis/label/val/'.format(dataset_name)
    hr_path = 'hr_f1_160'


    if not os.path.exists(label_train_dir):
       os.makedirs(label_train_dir)
    if not os.path.exists(label_val_dir):
       os.makedirs(label_val_dir)
    for fac in factor:
        train_path = main_path+ "factor_{}/train/".format(fac)
        val_path = main_path + "factor_{}/val/".format(fac)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        


    for fac in factor:
        x,y,z = data_hr.shape
        k=start
        for i in range(num_img):
            image_arr_cv = ut.normalize_image(data_hr[:,:,k])*255.
            img = ut.downsample_bicubic(image_arr_cv,factor=fac)
            name = 'lr_f1_160_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_image_cv(img,name,path)
    
    # saving lr images
    x,y,z = data_hr.shape
    k=start
    for i in range(num_img):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# ******************************************************************************************************************************

    # ## F2 145
    datapath_hr= '../OneDrive/AD_P522R_F2_145/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path ='hr_f2_145'


    for fac in factor:
        x,y,z = data_hr.shape
        k=start
        for i in range(num_img):
            image_arr_cv = ut.normalize_image(data_hr[:,:,k])*255.
            img = ut.downsample_bicubic(image_arr_cv,factor=fac)
            name = 'lr_f2_145_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_image_cv(img,name,path)
        
     
    # saving lr images
    x,y,z = data_hr.shape
    k=start
    for i in range(num_img):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# *********************************************************************************************************************************

# ## F4 149

# ## Validation set

    datapath_hr= '../OneDrive/AD_P522R_F4_149/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path = 'hr_f4_149'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start
        for i in range(num_img):
            image_arr_cv = ut.normalize_image(data_hr[:,:,k])*255.
            img = ut.downsample_bicubic(image_arr_cv,factor=fac)
            name = 'lr_f4_149_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/val/".format(fac)
            ut.save_image_cv(img,name,path)

    
    # saving lr images
    x,y,z = data_hr.shape
    k=start
    for i in range(num_img):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_val_dir)



# ***********************************************************************************************************************
# ## F5 153

    datapath_hr= '../OneDrive/AD_P522R_F5_153/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path ='hr_f5_153'


    for fac in factor:
        x,y,z = data_hr.shape
        k=start
        for i in range(num_img):
            image_arr_cv = ut.normalize_image(data_hr[:,:,k])*255.
            img = ut.downsample_bicubic(image_arr_cv,factor=fac)
            name = 'lr_f5_153_{}_z_{}'.format(fac,k)
            k+=1
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_image_cv(img,name,path)
   
    # saving lr images
    x,y,z = data_hr.shape
    k=start
    for i in range(num_img):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        k+=1
        ut.save_img_using_pil_lib(img,name,label_train_dir)


