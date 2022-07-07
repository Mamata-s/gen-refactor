import os

# # Generate 25UM dataset

# os.system("python generate_z_axis.py --dataset_size='MRI_25um'")
# os.system("python generate_full.py --dataset_size='MRI_25um'")

# # # Generate 50UM dataset
# os.system("python generate_z_axis.py --dataset_size='MRI_50um'")
# os.system("python generate_full.py --dataset_size='MRI_50um'")



# Generate train patch data from z-axis 25um images for factor 2 and 4
# factor=[2,4,6]
# dataset_name = 'resolution_dataset25'
# patch_size = 96
# stride = 95
# for fac in factor:
#     os.system(f"python create_patch.py --image-dir='../{dataset_name}/z_axis/factor_{fac}/train'  --label-dir='../{dataset_name}/z_axis/label/train' --patch-size={patch_size} --stride={stride} --output-image-dir='../{dataset_name}/patch/patch-{patch_size}/factor_{fac}/train/' --output-label-dir='../{dataset_name}/patch/patch-{patch_size}/label/train/'")
#     # for patch use val and test set from full z-axis images


# ***************************************************************************************************************************************************
# GENERATE GAUSSIAN DATASET


# Generate 25UM dataset

os.system("python generate_z_axis_gaussian.py --dataset_size='MRI_25um'")
# os.system("python generate_full_gaussian.py --dataset_size='MRI_25um'")

# # # Generate 50UM dataset
# os.system("python generate_z_axis_gaussian.py --dataset_size='MRI_50um'")
# os.system("python generate_full_gaussian.py --dataset_size='MRI_50um'")


# #Generate Patch Data
# # Generate train patch data from z-axis 50um images for factor 2
# os.system("python create_patch.py --image-dir='../gaussian_dataset50/z_axis/factor_2/train'  --label-dir='../gaussian_dataset50/z_axis/label/train' --patch-size=95 --num-patch=15 --output-image-dir='../gaussian_dataset50/patch/patch-95/factor_2/train/' --output-label-dir='../gaussian_dataset50/patch/patch-95/factor_2/label/train/'")
# # # Generate val patch data from z-axis 50um images for factor 2
# os.system("python create_patch.py --image-dir='../gaussian_dataset50/z_axis/factor_2/val'  --label-dir='../resolution_dataset50/z_axis/label/val' --patch-size=95 --num-patch=10 --output-image-dir='../gaussian_dataset50/patch/patch-64/factor_2/val/' --output-label-dir='../gaussian_dataset50/patch/patch-95/factor_2/label/val/'")



# #Generate Patch Data
# # Generate train patch data from z-axis 50um images for factor 4
# os.system("python create_patch.py --image-dir='../gaussian_dataset50/z_axis/factor_4/train'  --label-dir='../gaussian_dataset50/z_axis/label/train' --patch-size=95 --num-patch=15 --output-image-dir='../gaussian_dataset50/patch/patch-95/factor_4/train/' --output-label-dir='../gaussian_dataset50/patch/patch-95/factor_4/label/train/'")
# # # Generate val patch data from z-axis 50um images for factor 4
# os.system("python create_patch.py --image-dir='../gaussian_dataset50/z_axis/factor_4/val'  --label-dir='../gaussian_dataset50/z_axis/label/val' --patch-size=95 --num-patch=10 --output-image-dir='../gaussian_dataset50/patch/patch-95/factor_4/val/' --output-label-dir='../gaussian_dataset50/patch/patch-95/factor_4/label/val/'")



# ***************************************************************************************************************************************************************
# GENERATE BICUBIC DATASET

# os.system("python generate_z_axis_bicubic.py --dataset_size='MRI_25um'")
# os.system("python generate_full_bicubic.py --dataset_size='MRI_25um'")

# # # Generate 50UM dataset
# os.system("python generate_z_axis_bicubic.py --dataset_size='MRI_50um'")
# os.system("python generate_full_bicubic.py --dataset_size='MRI_50um'")


# #Generate Patch Data
# # Generate train patch data from z-axis 50um images for factor 2
# os.system("python create_patch.py --image-dir='../bicubic_dataset50/z_axis/factor_2/train'  --label-dir='../bicubic_dataset50/z_axis/label/train' --patch-size=95 --num-patch=15 --output-image-dir='../bicubic_dataset50/patch/patch-95/factor_2/train/' --output-label-dir='../bicubic_dataset50/patch/patch-95/factor_2/label/train/'")
# # # Generate val patch data from z-axis 50um images for factor 2
# os.system("python create_patch.py --image-dir='../bicubic_dataset50/z_axis/factor_2/val'  --label-dir='../bicubic_dataset50/z_axis/label/val' --patch-size=95 --num-patch=10 --output-image-dir='../bicubic_dataset50/patch/patch-64/factor_2/val/' --output-label-dir='../bicubic_dataset50/patch/patch-95/factor_2/label/val/'")



# #Generate Patch Data
# # Generate train patch data from z-axis 50um images for factor 4
# os.system("python create_patch.py --image-dir='../bicubic_dataset50/z_axis/factor_4/train'  --label-dir='../bicubic_dataset50/z_axis/label/train' --patch-size=95 --num-patch=15 --output-image-dir='../bicubic_dataset50/patch/patch-95/factor_4/train/' --output-label-dir='../bicubic_dataset50/patch/patch-95/factor_4/label/train/'")
# # # Generate val patch data from z-axis 50um images for factor 4
# os.system("python create_patch.py --image-dir='../bicubic_dataset50/z_axis/factor_4/val'  --label-dir='../bicubic_dataset50/z_axis/label/val' --patch-size=95 --num-patch=10 --output-image-dir='../bicubic_dataset50/patch/patch-95/factor_4/val/' --output-label-dir='../bicubic_dataset50/patch/patch-95/factor_4/label/val/'")

