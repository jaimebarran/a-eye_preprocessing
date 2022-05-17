import os
import dicom2nifti

input_directory = '../../../1_Soenke_Segmentation/DICOM_test/'
output_directory = '../../../1_Soenke_Segmentation/NIFTI_test/'
filename = 'T1_code.nii'

# Converting a directory with dicom files to nifti files
dicom2nifti.convert_directory(input_directory, output_directory)

# Converting a directory with only 1 series to 1 nifti file
# dicom2nifti.dicom_series_to_nifti(input_directory, output_directory+filename, reorient_nifti=True)