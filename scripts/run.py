import os

# Generate 25UM dataset

os.system("python generate_z_axis.py --dataset_size='MRI_25um'")
os.system("python generate_full.py --dataset_size='MRI_25um'")

# # Generate 50UM dataset
os.system("python generate_z_axis.py --dataset_size='MRI_50um'")
os.system("python generate_full.py --dataset_size='MRI_50um'")


#Generate Patch Data
# Generate train patch data from z-axis 50um images for factor 2
os.system("python create_patch.py --image-dir='../resolution_dataset50/z_axis/factor_2/train'  --label-dir='../resolution_dataset50/z_axis/label/train' --patch-size=64 --num-patch=10 --output-image-dir='../resolution_dataset50/patch/patch-64/factor_2/train/' --output-label-dir='../resolution_dataset50/patch/patch-64/factor_2/label/train/'")
# # Generate val patch data from z-axis 50um images for factor 2
os.system("python create_patch.py --image-dir='../resolution_dataset50/z_axis/factor_2/val'  --label-dir='../resolution_dataset50/z_axis/label/val' --patch-size=64 --num-patch=10 --output-image-dir='../resolution_dataset50/patch/patch-64/factor_2/val/' --output-label-dir='../resolution_dataset50/patch/patch-64/factor_2/label/val/'")



