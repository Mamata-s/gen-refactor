import utility as ut
import os
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hr-image-dir', type=str)
    parser.add_argument('--dataset_size',type=str,required=False,default='MRI_25um')
    args = parser.parse_args()


    factor = [2]

    dataset_size = args.dataset_size
    
    if dataset_size =='MRI_25um':
        dataset_name = 'resolution_dataset25'
        start_z = 20
        end_z = 260

        start_x = 0
        end_x = 620

        start_y=60
        end_y = 470
    else:
       dataset_name = 'resolution_dataset50' 
       start_z = 42
       end_z= 112

       start_x = 0
       end_x= 321

       start_y=42
       end_y =223


    num_z = end_z-start_z+1
    num_x= end_x-start_x+1
    num_y = end_y-start_y+1

    datapath_hr = '../OneDrive/AD_P522R_F1_160/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    main_path= '../{}/full/'.format(dataset_name)
    label_train_dir = '../{}/full/label/train/'.format(dataset_name)
    label_val_dir ='../{}/full/label/val/'.format(dataset_name)


    if not os.path.exists(label_train_dir):
        os.makedirs(label_train_dir)
    if not os.path.exists(label_val_dir):
        os.makedirs(label_val_dir)

    factor=[2,4,6,8]
    for fac in factor:
        train_path = main_path+ "factor_{}/train/".format(fac)
        val_path = main_path+ "factor_{}/val/".format(fac)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)


    # # Z-axis Images
    # ### F1 160

    hr_path ='hr_f1_160'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_z
        for i in range(num_z):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f1_160_{}_z_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1
    
        
    # saving label images
    x,y,z = data_hr.shape
    k=start_z
    for i in range(num_z):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1

    # *****************************************************************************************************************************

    # ## F2 145
    datapath_hr= '../OneDrive/AD_P522R_F2_145/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f2_145'



    for fac in factor:
        x,y,z = data_hr.shape
        k=start_z
        for i in range(num_z):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f2_145_{}_z_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1
    
    # saving label images
    x,y,z = data_hr.shape
    k=start_z
    for i in range(num_z):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1

    # ********************************************************************************************************************************
    # ## F4 149

    # ## Validation set

    datapath_hr= '../OneDrive/AD_P522R_F4_149/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    hr_path ='hr_f4_149'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_z
        for i in range(num_z):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f4_149_{}_z_{}'.format(fac,k)
            save_path = main_path+'factor_{}/val/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1
    
        
    # saving label images
    x,y,z = data_hr.shape
    k=start_z
    for i in range(num_z):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        ut.save_img_using_pil_lib(img,name,label_val_dir)
        k+=1

    # *******************************************************************************************************************************


    # ## F5 153

    datapath_hr= '../OneDrive/AD_P522R_F5_153/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f5_153'


    for fac in factor:
        x,y,z = data_hr.shape
        k=start_z
        for i in range(num_z):
            img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f5_153_{}_z_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)  
            k+=1 
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_z
    for i in range(num_z):
        name= hr_path+ '_z_'+str(k)
        img = ut.normalize_image(data_hr[:,:,k])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1

    # ______________***********************____________________************************___________********************************************************************************


    # # Yaxis Images

    # ## F1 160


    datapath_hr= '../OneDrive/AD_P522R_F1_160/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f1_160'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_y
        for i in range(num_y):
            img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f1_160_{}_y_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1

    # saving label images
    x,y,z = data_hr.shape
    k=start_y
    for i in range(num_y):
        name= hr_path+ '_y_'+str(k)
        img = ut.normalize_image(data_hr[:,k,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1


    # ************************************************************************************************************************

    # ## F2 145

    datapath_hr= '../OneDrive/AD_P522R_F2_145/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

   
    hr_path ='hr_f2_145'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_y
        for i in range(num_y):
            img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f2_145_{}_y_{}'.format(fac,k)
            save_path = main_path+'factor_{}/val/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1

    
    # saving lr images
    x,y,z = data_hr.shape
    k=start_y
    for i in range(num_y):
        name= hr_path+ '_y_'+str(k)
        img = ut.normalize_image(data_hr[:,k,:])
        ut.save_img_using_pil_lib(img,name,label_val_dir)
        k+=1

    # ******************************************************************************************************************************

    # ## F4 149

    datapath_hr= '../OneDrive/AD_P522R_F4_149/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f4_149'


    for fac in factor:
        x,y,z = data_hr.shape
        k=start_y
        for i in range(num_y):
            img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f4_149_{}_y_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1 
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_y
    for i in range(num_y):
        name= hr_path+ '_y_'+str(k)
        img = ut.normalize_image(data_hr[:,k,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1


    # *********************************************************************************************************************************

    # ## F5 153


    datapath_hr= '../OneDrive/AD_P522R_F5_153/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    hr_path ='hr_f5_153'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_y
        for i in range(num_y):
            img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f5_153_{}_y_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)       
            k+=1
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_y
    for i in range(num_y):
        name= hr_path+ '_y_'+str(k)
        img = ut.normalize_image(data_hr[:,k,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1

    # *******************************************************************************************************************************

    # ## X axis Images

    # ## F1 160


    datapath_hr= '../OneDrive/AD_P522R_F1_160/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    hr_path ='hr_f1_160'

    for fac in factor:
        x,y,z = data_hr.shape
        k=start_x
        for i in range(num_x):
            img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f1_160_{}_x_{}'.format(fac,k)
            save_path = main_path+'factor_{}/val/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path) 
            k+=1 
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_x
    for i in range(num_x):
        name= hr_path+ '_x_'+str(k)
        img = ut.normalize_image(data_hr[k,:,:])
        ut.save_img_using_pil_lib(img,name,label_val_dir)
        k+=1

    # ********************************************************************************************************************************

    # ## F2 145

    datapath_hr= '../OneDrive/AD_P522R_F2_145/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f2_145'
    


    for fac in factor:
        x,y,z = data_hr.shape
        k=start_x
        for i in range(num_x):
            img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f2_145_{}_x_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)   
            k+=1
        
    # saving label images
    x,y,z = data_hr.shape
    k=start_x
    for i in range(num_x):
        name= hr_path+ '_x_'+str(k)
        img = ut.normalize_image(data_hr[k,:,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k +=1
    # *********************************************************************************************************************************

    # ## F4 149

    datapath_hr= '../OneDrive/AD_P522R_F4_149/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f4_149'

    
    for fac in factor:
        x,y,z = data_hr.shape
        k=start_x
        for i in range(num_x):
            img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f4_149_{}_x_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path)
            k+=1 
        
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_x
    for i in range(num_x):
        name= hr_path+ '_x_'+str(k)
        img = ut.normalize_image(data_hr[k,:,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1
    # ********************************************************************************************************************************

    # ## F5 153

    datapath_hr= '../OneDrive/AD_P522R_F5_153/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    hr_path ='hr_f5_153'


    for fac in factor:
        x,y,z = data_hr.shape
        k=start_x
        for i in range(num_x):
            img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name= 'lr_f5_153_{}_x_{}'.format(fac,k)
            save_path = main_path+'factor_{}/train/'.format(fac)
            ut.save_img_using_pil_lib(img,name,save_path) 
            k+=1
        
        
    # saving lr images
    x,y,z = data_hr.shape
    k=start_x
    for i in range(num_x):
        name= hr_path+ '_x_'+str(k)
        img = ut.normalize_image(data_hr[k,:,:])
        ut.save_img_using_pil_lib(img,name,label_train_dir)
        k+=1





