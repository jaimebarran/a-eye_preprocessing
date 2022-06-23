'''
Align T1 image to labels' header
'''

import nibabel as nb
from pathlib import Path
import numpy as np
import glob, os

base_dir = '/mnt/sda1/ANTs/a123/'

for folder1 in os.listdir(base_dir):
    # # t1 = nb.load("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_T1.nii.gz")
    # t1 = nb.load(base_dir+folder1+'/input/'+folder1+'_T1.nii.gz')
    # # segments = [nb.load("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_labels.nii.gz")]
    # segments = nb.load(base_dir+folder1+'/input/'+folder1+'_labels.nii.gz')
    # header = segments.header.copy()
    # header.set_data_dtype("uint8")
    # nii = nb.Nifti1Image(t1.dataobj, segments.affine, header)
    # # nii.to_filename("/mnt/sda1/ANTs/a123/sub-01/input/sub-01_T1_test.nii.gz")
    # nii.to_filename(base_dir+folder1+'/input/'+folder1+'_T1_oriented.nii.gz')
    
    # Dealing with files in that folder
    for f in glob.glob(base_dir+folder1+'/input/'+folder1+'_T1_oriented_cropped.nii.gz'):
        os.remove(f)