import utility as ut


# # Z-axis Images
# ### F1 160

datapath_hr = '../OneDrive/AD_P522R_F1_160/MRI_25um/f1_25.nii'
data_hr = ut.load_data_nii(datapath_hr)

factor = [2,4,6,8,10]

data_or = ut.normalize_image(data_hr[:,:,65])
ut.save_img_using_pil_lib(data_or,'original','../')

x,y,z = data_hr.shape

for i in factor:
    img = ut.crop_pad_kspace(data_hr[:,:,65],pad=True,factor=i)
    img = ut.normalize_image(img)
    name = "downsampled_fac_"+str(i)
    ut.save_img_using_pil_lib(img,name,'../')
