import numpy as np
import os,sys
import nibabel as nib
import SimpleITK as sitk
import pandas as pd

# Paths
main_path = '/mnt/sda1/Repos/a-eye/Data/SHIP_dataset/'
input_image_path = main_path + 'non_labeled_dataset_nifti/' # TODO: change to non_labeled_dataset_nifti_reg
input_label_path = main_path + 'non_labeled_dataset_nifti_cropped_reg/'
output_path = main_path + 'non_labeled_dataset_nifti_cropped/'

i=0
for folder1 in sorted(os.listdir(input_image_path)):
    image_path = input_image_path + folder1 + '/' + folder1 + '.nii.gz' # image
    labels_path = input_label_path + folder1 + '/labels.nii.gz' # labels
    bound = 15 # boundary for the bounding box (margins)

    image = sitk.ReadImage(image_path)
    all_segments = sitk.ReadImage(labels_path)
    image_x_size, image_y_size, image_z_size = image.GetSize()
    print(f"image_x_size {image_x_size} image_y_size {image_y_size} image_z_size{image_z_size}")

    # Mask
    all_segments_mask = all_segments > 0
    # sitk.WriteImage(all_segments_mask, base_dir+folder+'/input/'+folder+'_labels_mask.nii.gz')

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
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sitk.WriteImage(image_crop, output_path + folder1 + '_cropped.nii.gz')

    i+=1
    if (i==10):
        break     