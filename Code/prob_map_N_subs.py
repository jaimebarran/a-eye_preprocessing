from shutil import which
import nibabel as nb
import numpy as np
import SimpleITK as sitk
import os
from pathlib import Path
from scipy.stats import mode

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/'
output_dir = base_dir+'Probability_Maps/Per_Class/'
# Create output directories
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# template_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0.nii.gz'
# mask_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/all_segments_mask.nii.gz'
template_cropped_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/template0_cropped_15vox.nii.gz'
mask_cropped_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/all_segments_mask_cropped.nii.gz'

best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9
num_subjects = len(best_subjects_cc) # number of subjects
threshold = 2/num_subjects # to compute the probabilities

im_template = sitk.ReadImage(template_cropped_path)
im_mask = sitk.ReadImage(mask_cropped_path)

# Subjects' labels
segments = [nb.load(f) for f in Path(base_dir).rglob("reg_cropped_best_subjects/*/*labels2template.nii.gz")]
# print(segments[0].get_fdata()[0,0,0])
header = segments[0].header.copy()
header.set_data_dtype("uint8")

# Matrix of zeros of the size of the image
# matrix = np.zeros_like(segments[0].dataobj, dtype="uint8")
prob_matrix_lens = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_globe = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_nerve = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_int_fat = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_ext_fat = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_lat_mus = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_med_mus = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_inf_mus = np.zeros_like(segments[0].dataobj, dtype="float")
prob_matrix_sup_mus = np.zeros_like(segments[0].dataobj, dtype="float")

# Bounding box
lsif = sitk.LabelStatisticsImageFilter() # It requires intensity and label images
lsif.Execute(im_template, im_mask) # Mask! Where all the labels are 1!
bounding_box = np.array(lsif.GetBoundingBox(1)) # GetBoundingBox(label)
print(f"Bounding box:  {bounding_box}") # [xmin, xmax, ymin, ymax, zmin, zmax]

# Loop
for x in range(bounding_box[0], bounding_box[1]+1):
    for y in range(bounding_box[2], bounding_box[3]+1):
        for z in range(bounding_box[4], bounding_box[5]+1):
            arr = np.zeros(num_subjects) # N subjects
            for i in range(num_subjects):
                arr[i] = segments[i].get_fdata()[x,y,z] # Array of votes from each subject for specific point [x,y,z]
                # if arr[i] == 1: print(f'arr[{i}==1!]')
            prob = np.zeros(9) # 9 classes
            if np.any(arr): # check if array has any nonzero value
                for j in range(len(prob)):
                    # if np.count_nonzero(arr == 1) > 0 : print(f'numpy has counted 1s in arr')
                    prob[j] = np.count_nonzero(arr ==  j+1) / len(arr) # Array of probabilities for each class
                if prob[0] > 0: print(f'there is lens!')
                prob_matrix_lens[x,y,z] = prob[0] # np.interp(prob[0], [0,1], [0,9])
                prob_matrix_globe[x,y,z] = prob[1]
                prob_matrix_nerve[x,y,z] = prob[2]
                prob_matrix_int_fat[x,y,z] = prob[3]
                prob_matrix_ext_fat[x,y,z] = prob[4]
                prob_matrix_lat_mus[x,y,z] = prob[5]
                prob_matrix_med_mus[x,y,z] = prob[6]
                prob_matrix_inf_mus[x,y,z] = prob[7]
                prob_matrix_sup_mus[x,y,z] = prob[8]

# Probability map representation
nii_lens = nb.Nifti1Image(prob_matrix_lens, segments[0].affine, header)
nii_globe = nb.Nifti1Image(prob_matrix_globe, segments[0].affine, header)
nii_nerve = nb.Nifti1Image(prob_matrix_nerve, segments[0].affine, header)
nii_int_fat = nb.Nifti1Image(prob_matrix_int_fat, segments[0].affine, header)
nii_ext_fat = nb.Nifti1Image(prob_matrix_ext_fat, segments[0].affine, header)
nii_lat_mus = nb.Nifti1Image(prob_matrix_lat_mus, segments[0].affine, header)
nii_med_mus = nb.Nifti1Image(prob_matrix_med_mus, segments[0].affine, header)
nii_inf_mus = nb.Nifti1Image(prob_matrix_inf_mus, segments[0].affine, header)
nii_sup_mus = nb.Nifti1Image(prob_matrix_sup_mus, segments[0].affine, header)
nii_lens.to_filename(output_dir+"prob_map_cropped_preMaxAPost_lens.nii.gz")
nii_globe.to_filename(output_dir+"prob_map_cropped_preMaxAPost_globe.nii.gz")
nii_nerve.to_filename(output_dir+"prob_map_cropped_preMaxAPost_nerve.nii.gz")
nii_int_fat.to_filename(output_dir+"prob_map_cropped_preMaxAPost_int_fat.nii.gz")
nii_ext_fat.to_filename(output_dir+"prob_map_cropped_preMaxAPost_ext_fat.nii.gz")
nii_lat_mus.to_filename(output_dir+"prob_map_cropped_preMaxAPost_lat_mus.nii.gz")
nii_med_mus.to_filename(output_dir+"prob_map_cropped_preMaxAPost_med_mus.nii.gz")
nii_inf_mus.to_filename(output_dir+"prob_map_cropped_preMaxAPost_inf_mus.nii.gz")
nii_sup_mus.to_filename(output_dir+"prob_map_cropped_preMaxAPost_sup_mus.nii.gz")