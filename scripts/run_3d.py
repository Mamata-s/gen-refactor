import os


factor_list = [2]
# #creating training set for factor 2  and 4   (with strie 20 generates 378 cubes of size 96*96*96)                                                                                                                                                                                                                                                                                                                                                                                      
# patch_size =96
# stride = 20
# factor=2
# data_dir='../OneDrive/AD_P522R_F1_160/MRI_50um/mag_sos_wn.nii'
# name = 'f1_160'
# output_image_dir ='../3d_dataset50/factor_2/train/'
# output_label_dir ='../3d_dataset50/label/train/'
# os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")


# F1_160 
#creating training set for factor 2  and 4   (with strie 40 generates 120 cubes of size 64*64*64)   from 

for fac in factor_list:
    patch_size = 64
    stride = 40
    factor=fac
    data_dir='../OneDrive/AD_P522R_F1_160/MRI_50um/mag_sos_wn.nii'
    name = 'f1_160'
    output_image_dir ='../3d_dataset50/factor_{}/train/'.format(fac)
    output_label_dir ='../3d_dataset50/label/train/'
    os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")


# F2_145
# creating training set for factor 2  and 4   (with strie 20 generates 378 cubes of size 96*96*96)   from 

for fac in factor_list:
    patch_size =64
    stride = 40
    factor=fac
    data_dir='../OneDrive/AD_P522R_F2_145/MRI_50um/mag_sos_wn.nii'
    name = 'f2_145'
    output_image_dir ='../3d_dataset50/factor_{}/train/'.format(fac)
    output_label_dir ='../3d_dataset50/label/train/'
    os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")


# # F3_147
# #creating training set for factor 2  and 4   (with strie 20 generates 378 cubes of size 96*96*96)   from 

for fac in factor_list:
    patch_size =64
    stride = 40
    factor=fac
    data_dir='../OneDrive/AD_P522R_F3_147/MRI_50um/mag_sos_wn.nii'
    name = 'f3_147'
    output_image_dir ='../3d_dataset50/factor_{}/train/'.format(fac)
    output_label_dir ='../3d_dataset50/label/train/'
    os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")


# # F4_149
# #creating training set for factor 2  and 4   (with strie 20 generates 378 cubes of size 96*96*96)   from 

for fac in factor_list:
    patch_size =64
    stride = 40
    factor=fac
    data_dir='../OneDrive/AD_P522R_F4_149/MRI_50um/mag_sos_wn.nii'
    name = 'f4_149'
    output_image_dir ='../3d_dataset50/factor_{}/train/'.format(fac)
    output_label_dir ='../3d_dataset50/label/train/'
    os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")


# # F5_153
# #creating VALIDATION set for factor 2  and 4   (with strie 20 generates 378 cubes of size 96*96*96)   from 

for fac in factor_list:
    patch_size =64
    stride = 40
    factor=fac
    data_dir='../OneDrive/AD_P522R_F5_153/MRI_50um/mag_sos_wn.nii'
    name = 'f5_153'
    output_image_dir ='../3d_dataset50/factor_{}/val/'.format(fac)
    output_label_dir ='../3d_dataset50/label/val/'
    os.system(f"python create_3d_patch.py --data-dir={data_dir} --factor={factor} --name={name} --patch-size={patch_size} --stride={stride} --output-image-dir={output_image_dir} --output-label-dir={output_label_dir}")
