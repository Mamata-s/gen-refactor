#added test set for z axis image for 25 micron only
import utility as ut
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size',type=str,required=False,default='MRI_25um')
    args = parser.parse_args()

    factor = [2,4,6]

    dataset_size = args.dataset_size
    
    if dataset_size =='MRI_25um':
        dataset_name = 'resolution_dataset25'
        start = 20
        end= 260
        num_img = end-start
        # genrating the test set index list and val set index list 

        train_list = [20,23,25,32,33,39,40,41,42,43,45,46,57,60,61,65,71,72,73,75,78,80,81,82,84,95,106,113,123,126,160,184,207,208,
        210,211,212,213,214,215,216,217,227,230,232,236,238,240,250,252]
        
        test_list= [42,71,88,95,96,103,133,141,147,154,167,168,181,182,183,191,200,207,212,214,224,231,240,247,248]

        original_list = [x for x in range(20,260)]
    else:
       dataset_name = 'resolution_dataset50' 
       start = 42
       end = 112
       num_img = end-start

       test_list =[46, 48, 55, 56, 59, 63, 69, 70, 72, 78, 90, 95, 96, 99, 102, 103, 104, 107, 112, 113]
       original_list = [x for x in range(42,113)]
    
    val_list = list(set(original_list) - set(test_list))

    # # Z-axis Images
    # ### F1 160
    datapath_hr = '../OneDrive/AD_P522R_F1_160/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)


    main_path= '../{}/z_axis/'.format(dataset_name)
    label_train_dir = '../{}/z_axis/label/train/'.format(dataset_name)
    label_val_dir ='../{}/z_axis/label/val/'.format(dataset_name)
    label_test_dir ='../{}/z_axis/label/test/'.format(dataset_name)
    hr_path = 'hr_f1_160'


    if not os.path.exists(label_train_dir):
       os.makedirs(label_train_dir)
    if not os.path.exists(label_val_dir):
       os.makedirs(label_val_dir)
    if not os.path.exists(label_test_dir):
        os.makedirs(label_test_dir)
    for fac in factor:
        train_path = main_path+ "factor_{}/train/".format(fac)
        val_path = main_path + "factor_{}/val/".format(fac)
        test_path = main_path + "factor_{}/test/".format(fac)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(val_path):
            os.makedirs(val_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        


    for fac in factor:
        for index in train_list:
            img = ut.crop_pad_kspace(data_hr[:,:,index],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f1_160_{}_z_{}'.format(fac,index)
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
    
    # saving lr images
    for index in train_list:
        name= hr_path+ '_z_'+str(index)
        img = ut.normalize_image(data_hr[:,:,index])
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# ******************************************************************************************************************************

    # ## F2 145
    datapath_hr= '../OneDrive/AD_P522R_F2_145/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path ='hr_f2_145'


    for fac in factor:
        for index in train_list:
            img = ut.crop_pad_kspace(data_hr[:,:,index],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f2_145_{}_z_{}'.format(fac,index)
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
     
    # saving lr images
    for index in train_list:
        name= hr_path+ '_z_'+str(index)
        img = ut.normalize_image(data_hr[:,:,index])
        ut.save_img_using_pil_lib(img,name,label_train_dir)

# *********************************************************************************************************************************

# ## F4 149

# ## Validation set

    datapath_hr= '../OneDrive/AD_P522R_F4_149/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path = 'hr_f4_149'

    # saving val set
    for fac in factor:
        for index in val_list:
            img = ut.crop_pad_kspace(data_hr[:,:,index],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f4_149_{}_z_{}'.format(fac,index)
            path = main_path+"factor_{}/val/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)

    
    # saving hr images for val set

    for index in val_list:
        name= hr_path+ '_z_'+str(index)
        img = ut.normalize_image(data_hr[:,:,index])
        print('saving image')
        print(ut.save_img_using_pil_lib(img,name,label_val_dir))

     # saving test set
    for fac in factor:
        for index in test_list:
            img = ut.crop_pad_kspace(data_hr[:,:,index],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f4_149_{}_z_{}'.format(fac,index)
            path = main_path+"factor_{}/test/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)

    
    # saving hr images for test set
    for index in test_list:
        name= hr_path+ '_z_'+str(index)
        img = ut.normalize_image(data_hr[:,:,index])
        ut.save_img_using_pil_lib(img,name,label_test_dir)



# ***********************************************************************************************************************
# ## F5 153

    datapath_hr= '../OneDrive/AD_P522R_F5_153/{}/mag_sos_wn.nii'.format(dataset_size)
    data_hr = ut.load_data_nii(datapath_hr)

    main_path= '../{}/z_axis/'.format(dataset_name)
    hr_path ='hr_f5_153'


    for fac in factor:
        for index in train_list:
            img = ut.crop_pad_kspace(data_hr[:,:,index],pad=True,factor=fac)
            img = ut.normalize_image(img)
            name = 'lr_f5_153_{}_z_{}'.format(fac,index)
            path = main_path+"factor_{}/train/".format(fac)
            ut.save_img_using_pil_lib(img,name,path)
   
    # saving lr images
    for index in train_list:
        name= hr_path+ '_z_'+str(index)
        img = ut.normalize_image(data_hr[:,:,index])
        ut.save_img_using_pil_lib(img,name,label_train_dir)


