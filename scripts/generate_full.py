import utility as ut
import os


# # Z-axis Images
# ### F1 160

datapath_hr = '../OneDrive_1_4-8-2022/AD_P522R_F1_160/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
label_train_dir = '../dataset/full/label/train/'
label_val_dir ='../dataset/full/label/val/'


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


factor=[2,4,6,8]
hr_path ='hr_f1_160'

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f1_160_{}_z_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1
   
    
# saving label images
x,y,z = data_hr.shape
k=42
for i in range(71):
    name= hr_path+ '_z_'+str(k)
    img = ut.normalize_image(data_hr[:,:,k])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1

# *****************************************************************************************************************************

# ## F2 145
datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F2_145/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f2_145'
factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f2_145_{}_z_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1
   
# saving label images
x,y,z = data_hr.shape
k=42
for i in range(71):
    name= hr_path+ '_z_'+str(k)
    img = ut.normalize_image(data_hr[:,:,k])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1

# ********************************************************************************************************************************
# ## F4 149

# ## Validation set

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F4_149/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f4_149'

factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f4_149_{}_z_{}'.format(fac,k)
        save_path = main_path+'factor_{}/val/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1
   
    
# saving label images
x,y,z = data_hr.shape
k=42
for i in range(71):
    name= hr_path+ '_z_'+str(k)
    img = ut.normalize_image(data_hr[:,:,k])
    ut.save_img_using_pil_lib(img,name,label_val_dir)
    k+=1

# *******************************************************************************************************************************


# ## F5 153

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f5_153'

factor=[2,4,6,8]

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(71):
        img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f5_153_{}_z_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)  
        k+=1 
    
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(71):
    name= hr_path+ '_z_'+str(k)
    img = ut.normalize_image(data_hr[:,:,k])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1

# ______________***********************____________________************************___________********************************************************************************


# # Yaxis Images

# ## F1 160


datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F1_160/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'

factor=[2,4,6,8]
hr_path ='hr_f1_160'

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(182):
        img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f1_160_{}_y_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1

# saving label images
x,y,z = data_hr.shape
k=42
for i in range(182):
    name= hr_path+ '_y_'+str(k)
    img = ut.normalize_image(data_hr[:,k,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1


# ************************************************************************************************************************

# ## F2 145

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F2_145/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f2_145'
factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(182):
        img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f2_145_{}_y_{}'.format(fac,k)
        save_path = main_path+'factor_{}/val/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1

  
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(182):
    name= hr_path+ '_y_'+str(k)
    img = ut.normalize_image(data_hr[:,k,:])
    ut.save_img_using_pil_lib(img,name,label_val_dir)
    k+=1

# ******************************************************************************************************************************

# ## F4 149

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F4_149/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f4_149'

factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(182):
        img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f4_149_{}_y_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1 
    
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(182):
    name= hr_path+ '_y_'+str(k)
    img = ut.normalize_image(data_hr[:,k,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1


# *********************************************************************************************************************************

# ## F5 153


datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f5_153'

factor=[2,4,6,8]

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(182):
        img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f5_153_{}_y_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)       
        k+=1
    
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(182):
    name= hr_path+ '_y_'+str(k)
    img = ut.normalize_image(data_hr[:,k,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1

# *******************************************************************************************************************************

# ## X axis Images

# ## F1 160


datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F1_160/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)


main_path= '../dataset/full/'

factor=[2,4,6,8]
hr_path ='hr_f1_160'

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(321):
        img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f1_160_{}_x_{}'.format(fac,k)
        save_path = main_path+'factor_{}/val/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path) 
        k+=1 
    
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(321):
    name= hr_path+ '_x_'+str(k)
    img = ut.normalize_image(data_hr[k,:,:])
    ut.save_img_using_pil_lib(img,name,label_val_dir)
    k+=1

# ********************************************************************************************************************************

# ## F2 145

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F2_145/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)


main_path= '../dataset/full/'
hr_path ='hr_f2_145'
factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(321):
        img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f2_145_{}_x_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)   
        k+=1
    
# saving label images
x,y,z = data_hr.shape
k=42
for i in range(321):
    name= hr_path+ '_x_'+str(k)
    img = ut.normalize_image(data_hr[k,:,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k +=1
# *********************************************************************************************************************************

# ## F4 149

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F4_149/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f4_149'

factor=[2,4,6,8]


for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(321):
        img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f4_149_{}_x_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path)
        k+=1 
    
    
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(321):
    name= hr_path+ '_x_'+str(k)
    img = ut.normalize_image(data_hr[k,:,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1
# ********************************************************************************************************************************

# ## F5 153

datapath_hr= '../OneDrive_1_4-8-2022/AD_P522R_F5_153/MRI_25um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../dataset/full/'
hr_path ='hr_f5_153'

factor=[2,4,6,8]

for fac in factor:
    x,y,z = data_hr.shape
    k=42
    for i in range(321):
        img = ut.crop_pad_kspace(data_hr[k,:,:],pad=True,factor=fac)
        img = ut.normalize_image(img)
        name= 'lr_f5_153_{}_x_{}'.format(fac,k)
        save_path = main_path+'factor_{}/train/'.format(fac)
        ut.save_img_using_pil_lib(img,name,save_path) 
        k+=1
    
      
# saving lr images
x,y,z = data_hr.shape
k=42
for i in range(321):
    name= hr_path+ '_x_'+str(k)
    img = ut.normalize_image(data_hr[k,:,:])
    ut.save_img_using_pil_lib(img,name,label_train_dir)
    k+=1





