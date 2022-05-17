from asyncore import loop
from fileinput import filename
import numpy
import os,sys
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
import numpy as np

# Steps
# 1. Load .nii images
# 2. Decompose .nii images into arrays
# 3. OR pixels (dilation)
# 4. Reconstruct .nii image

### Additional funtions ###
def loop_or(len):

    j = 0
    arr1 = img_array_norm['img%s' %0]
    arr2 = img_array_norm['img%s' %1]
    
    while j < len-1:
        output = np.logical_or(arr1, arr2)
        arr1 = output
        if j > (len-2):
            arr2 = img_array_norm['img%s' %(j+2)]
        j += 1
    return output
#############################  

input_directory = '../../../1_Soenke_Segmentation/NIFTI/'
output_directory = '../../../1_Soenke_Segmentation/Output/'
output_filename = 'fats.nii'

# File loop
i = 0 # to name each read image
img_array_norm = {}
for filename in os.listdir(input_directory):
    f = os.path.join(input_directory, filename)
    if os.path.isfile(f) and f.endswith('.nii'): # check if it is a .nii file
        # print(filename)
        img = sitk.ReadImage(f)
        img_array = sitk.GetArrayFromImage(img) # 2. Decomposition into arrays
        # img_array = sitk.GetArrayViewFromImage(img)
        minVal = np.amin(img_array)
        maxVal = np.amax(img_array)
        print(f"{filename}: {img_array.shape}")
        print(f"Intensities: min = {minVal}, max = {maxVal} \n")
        img_array_norm['img%s' %i] = img_array / maxVal # normalized intensities
        # print(np.count_nonzero(img_array_norm['img%s' %i] == 1)) # How many positive values the array has
    i += 1