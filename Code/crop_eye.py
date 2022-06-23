import numpy as np
import os,sys
import nibabel as nib
import SimpleITK as sitk
import pandas as pd

# Image
image_path = "/mnt/sda1/ANTs/a123/sub-18/input/"
image_filename = "sub-18_T1"
image_format = ".nii.gz"

# image_path = "/mnt/sda1/ANTs/input/mni152/"
# image_filename = "tpl-MNI152NLin2009cAsym_res-01_T1w"
# image_format = ".nii.gz"

# image_path = "/mnt/sda1/ANTs/input/colin27/"
# image_filename = "tpl-MNIColin27_T1w"
# image_format = ".nii.gz"

image_complete_path = image_path + image_filename + image_format

# Lables image
all_segments_path = "/mnt/sda1/ANTs/a123/sub-01/input/"
all_segments_filename = "sub-01_labels_oriented"
all_segments_format = ".nii.gz"

# all_segments_path = "/mnt/sda1/ANTs/a123/sub-01/output_mni152/"
# all_segments_filename = "all_segments_template"
# all_segments_format = ".nii.gz"

# all_segments_path = "/mnt/sda1/ANTs/a123/sub-01/output_colin27/"
# all_segments_filename = "all_segments_template"
# all_segments_format = ".nii.gz"

all_segments_complete_path = all_segments_path + all_segments_filename + all_segments_format

# Boundary for the bounding box
bound = 15

image = sitk.ReadImage(image_complete_path)
all_segments = sitk.ReadImage(all_segments_complete_path)
image_x_size, image_y_size, image_z_size = image.GetSize()
print(f"image_x_size {image_x_size} image_y_size {image_y_size} zimage_z_sizeSize {image_z_size}")

# Mask
all_segments_mask = all_segments > 0
sitk.WriteImage(all_segments_mask, all_segments_path + all_segments_filename + '_mask' + all_segments_format)

# Bounding box
lsif = sitk.LabelStatisticsImageFilter() # It requires intensity and label images
lsif.Execute(image, all_segments_mask) # Mask! Where all the labels are 1!
bounding_box = np.array(lsif.GetBoundingBox(1)) # GetBoundingBox(label)
print(f"Bounding box:  {bounding_box}") # [xmin, xmax, ymin, ymax, zmin, zmax]
bounding_box_expanded = bounding_box.copy()
bounding_box_expanded[0::2] -= bound # even indexes
bounding_box_expanded[1::2] += bound # odd indexes
print(f"Expanded bounding box: {bounding_box_expanded}")

# Limits
if bounding_box_expanded[0] < 0: bounding_box_expanded[0] = 0
if bounding_box_expanded[1] > image_x_size: bounding_box_expanded[1] = image_x_size
if bounding_box_expanded[2] < 0: bounding_box_expanded[2] = 0
if bounding_box_expanded[3] > image_y_size: bounding_box_expanded[3] = image_y_size
if bounding_box_expanded[4] < 0: bounding_box_expanded[4] = 0
if bounding_box_expanded[5] > image_z_size: bounding_box_expanded[5] = image_z_size
print(f"Expanded bounding box after limits: {bounding_box_expanded}")

# Crop
image_crop = image[int(bounding_box_expanded[0]):int(bounding_box_expanded[1]), # x
                   int(bounding_box_expanded[2]):int(bounding_box_expanded[3]), # y
                   int(bounding_box_expanded[4]):int(bounding_box_expanded[5])] # z
sitk.WriteImage(image_crop, all_segments_path + image_filename + "_cropped" + image_format)   
# The following is only needed with T1 image
all_segments_crop = all_segments[int(bounding_box_expanded[0]):int(bounding_box_expanded[1]), # x
                                 int(bounding_box_expanded[2]):int(bounding_box_expanded[3]), # y
                                 int(bounding_box_expanded[4]):int(bounding_box_expanded[5])] # z
sitk.WriteImage(all_segments_crop, all_segments_path + all_segments_filename + "_cropped" + all_segments_format)