'''
Align T1 image to labels' header
'''

import nibabel as nb
from pathlib import Path
import numpy as np
import glob, os

# base_dir = '/mnt/sda1/ANTs/a123/'
base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/' # for the custom template

# Allign origin to create custom template
segments = nb.load(base_dir+'sub-29/input/sub-29_labels.nii.gz')
header = segments.header.copy()
header.set_data_dtype("uint8")

for folder1 in os.listdir(base_dir):
    print(folder1)
    # t1 = nb.load("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_T1.nii.gz")
    t1 = nb.load(base_dir+folder1+'/input/'+folder1+'_T1.nii.gz')
    # segments = [nb.load("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_labels.nii.gz")]
    # segments = nb.load(base_dir+folder1+'/input/'+folder1+'_labels.nii.gz')
    # header = segments.header.copy()
    # header.set_data_dtype("uint8")

    # nii = nb.Nifti1Image(t1.dataobj, segments.affine, header)
    # # nii.to_filename("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_T1_test.nii.gz")
    # nii.to_filename(base_dir+folder1+'/input/'+folder1+'_T1_origin.nii.gz')

    # For the custom template
    nii = nb.Nifti1Image(t1.dataobj, t1.affine, t1.header)
    nii.set_sform(t1.affine, code=0)
    nii.set_qform(t1.affine, code=1)
    nii.to_filename(base_dir+folder1+'/input/'+folder1+'_T1_hdr.nii.gz')
    
    # Dealing with files in that folder
    # for f in glob.glob(base_dir+folder1+'/input/'+folder1+'_origin.nii.gz'):
    #     os.remove(f)