import utility as ut
from PIL import Image

''' generates the image with aroppping and bicubic upsampling
'''

# # Z-axis Images
# ### F1 160

datapath_hr = '../OneDrive/AD_P522R_F1_160/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/z_axis/f1_160/'
sub_path_2='lr_f1_160_2'
sub_path_4 ='lr_f1_160_4'
sub_path_8 ='lr_f1_160_8'
hr_path ='lr_f1_160'


# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_z_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     name= hr_path+ '_z_'+str(k)
#     img = ut.normalize_image(data_hr[:,:,k])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# ## F2 145
datapath_hr= '../OneDrive/AD_P522R_F2_145/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/z_axis/f2_145/'
sub_path_2='lr_f2_145_2'
sub_path_4 ='lr_f2_145_4'
sub_path_8 ='lr_f2_145_8'
hr_path ='lr_f2_145'


# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_z_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     name= hr_path+ '_z_'+str(k)
#     img = ut.normalize_image(data_hr[:,:,k])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# ## F4 149

# ## Validation set

datapath_hr= '../OneDrive/AD_P522R_F4_149/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/z_axis/f4_149/'
sub_path_2='lr_f4_149_2'
sub_path_4='lr_f4_149_4'
sub_path_8 ='lr_f4_149_8'
hr_path ='lr_f4_149'

  

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/val/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/val/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_z_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/val/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     name= hr_path+ '_z_'+str(k)
#     img = ut.normalize_image(data_hr[:,:,k])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/val/')


# ## F5 153


datapath_hr= '../OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/z_axis/f5_153/'
sub_path_2 ='lr_f5_153_2'
sub_path_4='lr_f5_153_4'
sub_path_8 ='lr_f5_153_8'
hr_path ='lr_f5_153'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(71):
    img = ut.crop_pad_kspace(data_hr[:,:,k],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_z_'+str(k)
    k+=1
    shape = (data_hr[:,:,k].shape[1],data_hr[:,:,k].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     img = ut.crop_pad_kspace(data_hr[:,:,k],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_z_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(71):
#     name= hr_path+ '_z_'+str(k)
#     img = ut.normalize_image(data_hr[:,:,k])
#     k+=1
    # ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# # Yaxis Images

# ## F1 160


datapath_hr= '../OneDrive/AD_P522R_F1_160/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/y_axis/f1_160/'
sub_path_2='lr_f1_160_2'
sub_path_4 ='lr_f1_160_4'
sub_path_8 ='lr_f1_160_8'
hr_path ='lr_f1_160'



# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_y_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     name= hr_path+ '_y_'+str(k)
#     img = ut.normalize_image(data_hr[:,k,:])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# ## F2 145


datapath_hr= '../OneDrive/AD_P522R_F2_145/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)


main_path= '../images/50um/y_axis/f2_145/'
sub_path_2='lr_f2_145_2'
sub_path_4 ='lr_f2_145_4'
sub_path_8 ='lr_f2_145_8'
hr_path ='lr_f2_145'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/val/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/val/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_y_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/val/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     name= hr_path+ '_y_'+str(k)
#     img = ut.normalize_image(data_hr[:,k,:])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/val/')


# ## F4 149

datapath_hr= '../OneDrive/AD_P522R_F4_149/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/y_axis/f4_149/'
sub_path_2='lr_f4_149_2'
sub_path_4='lr_f4_149_4'
sub_path_8 ='lr_f4_149_8'
hr_path ='lr_f4_149'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_y_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     name= hr_path+ '_y_'+str(k)
#     img = ut.normalize_image(data_hr[:,k,:])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# ## F5 153
datapath_hr= '../OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/y_axis/f5_153/'
sub_path_2 ='lr_f5_153_2'
sub_path_4='lr_f5_153_4'
sub_path_8 ='lr_f5_153_8'
hr_path ='lr_f5_153'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
k=42
for i in range(182):
    img = ut.crop_pad_kspace(data_hr[:,k,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_y_'+str(k)
    k+=1
    shape = (data_hr[:,k,:].shape[1],data_hr[:,k,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     img = ut.crop_pad_kspace(data_hr[:,k,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_y_'+str(k)
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# k=42
# for i in range(182):
#     name= hr_path+ '_y_'+str(k)
#     img = ut.normalize_image(data_hr[:,k,:])
#     k+=1
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')


# ## X axis Images

# ## F1 160

datapath_hr= '../OneDrive/AD_P522R_F1_160/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)


main_path= '../images/50um/x_axis/f1_160/'
sub_path_2='lr_f1_160_2'
sub_path_4 ='lr_f1_160_4'
sub_path_8 ='lr_f1_160_8'
hr_path ='lr_f1_160'


# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/val/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/val/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# for i in range(321):
#     img = ut.crop_pad_kspace(data_hr[i,:,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_x_'+str(i)
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/val/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# for i in range(321):
#     name= hr_path+ '_x_'+str(i)
#     img = ut.normalize_image(data_hr[i,:,:])
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/val/')


# ## F2 145

datapath_hr= '../OneDrive/AD_P522R_F2_145/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/x_axis/f2_145/'
sub_path_2='lr_f2_145_2'
sub_path_4 ='lr_f2_145_4'
sub_path_8 ='lr_f2_145_8'
hr_path ='lr_f2_145'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# for i in range(321):
#     img = ut.crop_pad_kspace(data_hr[i,:,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_x_'+str(i)
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# for i in range(321):
#     name= hr_path+ '_x_'+str(i)
#     img = ut.normalize_image(data_hr[i,:,:])
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')



# ## F4 149


datapath_hr= '../OneDrive/AD_P522R_F4_149/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/x_axis/f4_149/'
sub_path_2='lr_f4_149_2'
sub_path_4='lr_f4_149_4'
sub_path_8 ='lr_f4_149_8'
hr_path ='lr_f4_149'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# for i in range(321):
#     img = ut.crop_pad_kspace(data_hr[i,:,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_x_'+str(i)
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# for i in range(321):
#     name= hr_path+ '_x_'+str(i)
#     img = ut.normalize_image(data_hr[i,:,:])
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')

# ## F5 153

datapath_hr= '../OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii'
data_hr = ut.load_data_nii(datapath_hr)

main_path= '../images/50um/x_axis/f5_153/'
sub_path_2 ='lr_f5_153_2'
sub_path_4='lr_f5_153_4'
sub_path_8 ='lr_f5_153_8'
hr_path ='lr_f5_153'

# saving lr cropped and padded by fac 2
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=2)
    img = ut.normalize_image(img)
    name= sub_path_2 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_2/train/',upsample=True,shape=shape)
    
# saving lr cropped and padded by fac 4
x,y,z = data_hr.shape
for i in range(321):
    img = ut.crop_pad_kspace(data_hr[i,:,:],pad=False,factor=4)
    img = ut.normalize_image(img)
    name= sub_path_4 + '_x_'+str(i)
    shape = (data_hr[i,:,:].shape[1],data_hr[i,:,:].shape[0])
    ut.save_img_using_pil_lib(img,name,'../dataset/crop_factor_4/train/',upsample=True,shape=shape)
    
    
# # saving lr cropped and padded by fac 8
# x,y,z = data_hr.shape
# for i in range(321):
#     img = ut.crop_pad_kspace(data_hr[i,:,:],pad=True,factor=8)
#     img = ut.normalize_image(img)
#     name= sub_path_8 + '_x_'+str(i)
#     ut.save_img_using_pil_lib(img,name,'../dataset/factor_8/train/')    
    
# # saving lr images
# x,y,z = data_hr.shape
# for i in range(321):
#     name= hr_path+ '_x_'+str(i)
#     img = ut.normalize_image(data_hr[i,:,:])
#     ut.save_img_using_pil_lib(img,name,'../dataset/label/train/')