from asyncore import write
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
from sqlalchemy import true

# TODO: nDSC (normalized DSC)
def dice_norm_metric(ground_truth, predictions):
    '''
    For a single example returns DSC_norm, fpr, fnr
    '''

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0, 1.0, 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        fpr = fp / (len(gt.flatten()) - np.sum(gt))
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm # fpr, fnr

''' Data frame file generation
base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/' # {1, 5, 7, 9}
gt_path = base_dir + 'Probability_Maps/prob_map_cropped_th0.nii.gz' # GT

reader = sitk.ImageFileReader()
reader.SetFileName(gt_path)
gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
gt_arr = sitk.GetArrayFromImage(gt_sitk) # en numpy format

# List of best subjects
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9

# List of remaining subjects
all_subjects = list()
for i in range(35):
    all_subjects.append('sub-'+str(i+1).zfill(2))
rest_subjects = [elem for elem in all_subjects if elem not in best_subjects_cc]

# Save values in an array
# All labels
val_dsc = np.zeros(len(rest_subjects))
# val_hau = np.zeros(len(rest_subjects))
val_hau_avg = np.zeros(len(rest_subjects))
val_vol = np.zeros(len(rest_subjects))
val_ndsc = np.zeros(len(rest_subjects))
# Lens
val_dsc_lens = np.zeros(len(rest_subjects))
val_hau_avg_lens = np.zeros(len(rest_subjects))
val_vol_lens = np.zeros(len(rest_subjects))
val_ndsc_lens = np.zeros(len(rest_subjects))
# Globe
val_dsc_globe = np.zeros(len(rest_subjects))
val_hau_avg_globe = np.zeros(len(rest_subjects))
val_vol_globe = np.zeros(len(rest_subjects))
val_ndsc_globe = np.zeros(len(rest_subjects))
# Optic nerve
val_dsc_nerve = np.zeros(len(rest_subjects))
val_hau_avg_nerve = np.zeros(len(rest_subjects))
val_vol_nerve = np.zeros(len(rest_subjects))
val_ndsc_nerve = np.zeros(len(rest_subjects))
# Intraconal fat
val_dsc_int_fat = np.zeros(len(rest_subjects))
val_hau_avg_int_fat = np.zeros(len(rest_subjects))
val_vol_int_fat = np.zeros(len(rest_subjects))
val_ndsc_int_fat = np.zeros(len(rest_subjects))
# Extraconal fat
val_dsc_ext_fat = np.zeros(len(rest_subjects))
val_hau_avg_ext_fat = np.zeros(len(rest_subjects))
val_vol_ext_fat = np.zeros(len(rest_subjects))
val_ndsc_ext_fat = np.zeros(len(rest_subjects))
# Lateral rectus muscle
val_dsc_lat_mus = np.zeros(len(rest_subjects))
val_hau_avg_lat_mus = np.zeros(len(rest_subjects))
val_vol_lat_mus = np.zeros(len(rest_subjects))
val_ndsc_lat_mus = np.zeros(len(rest_subjects))
# Medial rectus muscle
val_dsc_med_mus = np.zeros(len(rest_subjects))
val_hau_avg_med_mus = np.zeros(len(rest_subjects))
val_vol_med_mus = np.zeros(len(rest_subjects))
val_ndsc_med_mus = np.zeros(len(rest_subjects))
# Inferior rectus muscle
val_dsc_inf_mus = np.zeros(len(rest_subjects))
val_hau_avg_inf_mus = np.zeros(len(rest_subjects))
val_vol_inf_mus = np.zeros(len(rest_subjects))
val_ndsc_inf_mus = np.zeros(len(rest_subjects))
# Superior rectus muscle
val_dsc_sup_mus = np.zeros(len(rest_subjects))
val_hau_avg_sup_mus = np.zeros(len(rest_subjects))
val_vol_sup_mus = np.zeros(len(rest_subjects))
val_ndsc_sup_mus = np.zeros(len(rest_subjects))
### Grouped labels ###
# # Fats
# val_dsc_fats = np.zeros(len(rest_subjects))
# val_hau_avg_fats = np.zeros(len(rest_subjects))
# val_vol_fats = np.zeros(len(rest_subjects))
# val_ndsc_fats = np.zeros(len(rest_subjects))
# # Muscles
# val_dsc_muscles = np.zeros(len(rest_subjects))
# val_hau_avg_muscles = np.zeros(len(rest_subjects))
# val_vol_muscles = np.zeros(len(rest_subjects))
# val_ndsc_muscles = np.zeros(len(rest_subjects))

for i in range(len(rest_subjects)):

    # Prediction image to compare to GT
    pr_path = base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2template.nii.gz' # Labels' image to compare to GT
    reader.SetFileName(pr_path)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    pr_arr = sitk.GetArrayFromImage(pr_sitk) # en numpy format 

    # ALL LABELS
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk, pr_sitk)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    # hausdorf_distance = hausdorf.GetHausdorffDistance()
    # val_hau[i] = hausdorf_distance
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr!=0, pr_arr!=0)
    val_ndsc[i] = nDSC

    # LENS
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==1, pr_sitk==1)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_lens[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_lens[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_lens[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==1, pr_arr==1)
    val_ndsc_lens[i] = nDSC
    
    # GLOBE EX LENS
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==2, pr_sitk==2)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_globe[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_globe[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_globe[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==2, pr_arr==2)
    val_ndsc_globe[i] = nDSC

    # OPTIC NERVE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==3, pr_sitk==3)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_nerve[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_nerve[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk, pr_sitk)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_nerve[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==3, pr_arr==3)
    val_ndsc_nerve[i] = nDSC

    # INTRACONAL FAT
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==4, pr_sitk==4)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_int_fat[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_int_fat[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==4, pr_sitk==4)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_int_fat[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==4, pr_arr==4)
    val_ndsc_int_fat[i] = nDSC

    # EXTRACONAL FAT
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==5, pr_sitk==5)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_ext_fat[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_ext_fat[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==5, pr_sitk==5)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_ext_fat[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==5, pr_arr==5)
    val_ndsc_ext_fat[i] = nDSC

    # LATERAL RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==6, pr_sitk==6)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_lat_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_lat_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==6, pr_sitk==6)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_lat_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==6, pr_arr==6)
    val_ndsc_lat_mus[i] = nDSC

    # MEDIAL RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==7, pr_sitk==7)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_med_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_med_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==7, pr_sitk==7)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_med_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==7, pr_arr==7)
    val_ndsc_med_mus[i] = nDSC

    # INFERIOR RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==8, pr_sitk==8)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_inf_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_inf_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==8, pr_sitk==8)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_inf_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==8, pr_arr==8)
    val_ndsc_inf_mus[i] = nDSC

    # SUPERIOR RECTUS MUSCLE
    # Measures Image Filter 
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==9, pr_sitk==9)
    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc_sup_mus[i] = dsc
    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol_sup_mus[i] = vol
    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==9, pr_sitk==9)
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg_sup_mus[i] = hausdorf_distance_avg
    # nDSC
    nDSC = dice_norm_metric(gt_arr==9, pr_arr==9)
    val_ndsc_sup_mus[i] = nDSC

    ### GROUPED LABELS ###
    # # FATS
    # # Measures Image Filter 
    # overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    # overlap_measures_filter.Execute(gt_sitk==4 or gt_sitk==5, pr_sitk==4 or pr_sitk==5)
    # # DSC
    # dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    # val_dsc_fats[i] = dsc
    # # Volume
    # vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    # val_vol_fats[i] = vol
    # # Hausdorff distance
    # hausdorf = sitk.HausdorffDistanceImageFilter()
    # hausdorf.Execute(gt_sitk==4 or gt_sitk==5, pr_sitk==4 or pr_sitk==5)
    # hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    # val_hau_avg_fats[i] = hausdorf_distance_avg
    # # nDSC
    # gt_mask = np.logical_or(gt_arr==4, gt_arr==5)
    # pr_mask = np.logical_or(pr_arr==4, pr_arr==5)
    # nDSC = dice_norm_metric(gt_mask, pr_mask)
    # val_ndsc_fats[i] = nDSC

    # # MUSCLES
    # # Measures Image Filter 
    # overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    # overlap_measures_filter.Execute(gt_sitk>=6 or gt_sitk<=9, pr_sitk>=6 or pr_sitk<=9)
    # # DSC
    # dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    # val_dsc_muscles[i] = dsc
    # # Volume
    # vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    # val_vol_muscles[i] = vol
    # # Hausdorff distance
    # hausdorf = sitk.HausdorffDistanceImageFilter()
    # hausdorf.Execute(gt_sitk>=6 or gt_sitk<=9, pr_sitk>=6 or pr_sitk<=9)
    # hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    # val_hau_avg_muscles[i] = hausdorf_distance_avg
    # # nDSC
    # gt_mask = np.logical_or.reduce((gt_arr==6, gt_arr==7, gt_arr==8, gt_arr==9))
    # pr_mask = np.logical_or.reduce((pr_arr==6, pr_arr==7, pr_arr==8, pr_arr==9))
    # nDSC = dice_norm_metric(gt_mask, pr_mask)
    # val_ndsc_muscles[i] = nDSC

# Save values to a csv
# metrics = ['Subject','DSC', 'Hausdorff', 'Hausdorff_avg']
# metrics = ['Subject', 'DSC_all', 'Haus_avg_all', 'Volume_all', 'DSC_lens', 'Haus_avg_lens', 'Volume_lens', 'DSC_globe', 'Haus_avg_globe', 'Volume_globe',
#             'DSC_nerve', 'Haus_avg_nerve', 'Volume_nerve', 'DSC_fats', 'Haus_avg_fats', 'Volume_fats', 'DSC_muscles', 'Haus_avg_muscles', 'Volume_muscles']
# metrics = ['Subject', 'DSC_all', 'Haus_avg_all', 'Volume_all', 'DSC_lens', 'Haus_avg_lens', 'Volume_lens', 'DSC_globe', 'Haus_avg_globe', 'Volume_globe',
#             'DSC_nerve', 'Haus_avg_nerve', 'Volume_nerve', 'DSC_int_fat', 'Haus_avg_int_fat', 'Volume_int_fat', 'DSC_ext_fat', 'Haus_avg_ext_fat', 'Volume_ext_fat',
#             'DSC_lat_mus', 'Haus_avg_lat_mus', 'Volume_lat_mus', 'DSC_med_mus', 'Haus_avg_med_mus', 'Volume_med_mus', 'DSC_inf_mus', 'Haus_avg_inf_mus', 'Volume_inf_mus',
#             'DSC_sup_mus', 'Haus_avg_sup_mus', 'Volume_sup_mus']
# metrics = ['Subject', 'DSC_all', 'Haus_avg_all', 'Volume_all', 'nDSC_all', 'DSC_lens', 'Haus_avg_lens', 'Volume_lens', 'nDSC_lens', 'DSC_globe', 'Haus_avg_globe', 
#             'Volume_globe', 'nDSC_globe', 'DSC_nerve', 'Haus_avg_nerve', 'Volume_nerve', 'nDSC_nerve', 'DSC_fats', 'Haus_avg_fats', 'Volume_fats', 'nDSC_fats',
#             'DSC_muscles', 'Haus_avg_muscles', 'Volume_muscles', 'nDSC_muscles']
metrics = ['Subject', 'DSC_all', 'Haus_avg_all', 'Volume_all', 'nDSC_all', 'DSC_lens', 'Haus_avg_lens', 'Volume_lens', 'nDSC_lens', 'DSC_globe', 'Haus_avg_globe', 'Volume_globe',
            'nDSC_globe', 'DSC_nerve', 'Haus_avg_nerve', 'Volume_nerve', 'nDSC_nerve', 'DSC_int_fat', 'Haus_avg_int_fat', 'Volume_int_fat', 'nDSC_int_fat', 'DSC_ext_fat',
            'Haus_avg_ext_fat', 'Volume_ext_fat', 'nDSC_ext_fat', 'DSC_lat_mus', 'Haus_avg_lat_mus', 'Volume_lat_mus', 'nDSC_lat_mus', 'DSC_med_mus', 'Haus_avg_med_mus',
            'Volume_med_mus', 'nDSC_med_mus', 'DSC_inf_mus', 'Haus_avg_inf_mus', 'Volume_inf_mus', 'nDSC_inf_mus', 'DSC_sup_mus', 'Haus_avg_sup_mus', 'Volume_sup_mus', 'nDSC_sup_mus']
# print(val_dsc[0], val_hau[0], val_hau_avg[0])
# vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol, val_dsc_lens, val_hau_avg_lens, val_vol_lens, val_dsc_globe, val_hau_avg_globe, val_vol_globe,
#                 val_dsc_nerve, val_hau_avg_nerve, val_vol_nerve, val_dsc_fats, val_hau_avg_fats, val_vol_fats, val_dsc_muscles, val_hau_avg_muscles, val_vol_muscles])
# vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol, val_dsc_lens, val_hau_avg_lens, val_vol_lens, val_dsc_globe, val_hau_avg_globe, val_vol_globe,
#                 val_dsc_nerve, val_hau_avg_nerve, val_vol_nerve, val_dsc_int_fat, val_hau_avg_int_fat, val_vol_int_fat, val_dsc_ext_fat, val_hau_avg_ext_fat, val_vol_ext_fat,
#                 val_dsc_lat_mus, val_hau_avg_lat_mus, val_vol_lat_mus, val_dsc_med_mus, val_hau_avg_med_mus, val_vol_med_mus, val_dsc_inf_mus, val_hau_avg_inf_mus, val_vol_inf_mus,
#                 val_dsc_sup_mus, val_hau_avg_sup_mus, val_vol_sup_mus])
# vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol, val_ndsc, val_dsc_lens, val_hau_avg_lens, val_vol_lens, val_ndsc_lens, val_dsc_globe, val_hau_avg_globe, 
#                 val_vol_globe, val_ndsc_globe, val_dsc_nerve, val_hau_avg_nerve, val_vol_nerve, val_ndsc_nerve, val_dsc_fats, val_hau_avg_fats, val_vol_fats, val_ndsc_fats,
#                 val_dsc_muscles, val_hau_avg_muscles, val_vol_muscles, val_ndsc_muscles])
vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol, val_ndsc, val_dsc_lens, val_hau_avg_lens, val_vol_lens, val_ndsc_lens, val_dsc_globe, val_hau_avg_globe, val_vol_globe,
                val_ndsc_globe, val_dsc_nerve, val_hau_avg_nerve, val_vol_nerve, val_ndsc_nerve, val_dsc_int_fat, val_hau_avg_int_fat, val_vol_int_fat, val_ndsc_int_fat, val_dsc_ext_fat,
                val_hau_avg_ext_fat, val_vol_ext_fat, val_ndsc_ext_fat, val_dsc_lat_mus, val_hau_avg_lat_mus, val_vol_lat_mus, val_ndsc_lat_mus, val_dsc_med_mus, val_hau_avg_med_mus,
                val_vol_med_mus, val_ndsc_med_mus, val_dsc_inf_mus, val_hau_avg_inf_mus, val_vol_inf_mus, val_ndsc_inf_mus, val_dsc_sup_mus, val_hau_avg_sup_mus, val_vol_sup_mus, val_ndsc_sup_mus])
vals = vals.T
# print(vals)
# print(f"type: {vals.dtype}, shape: {vals.shape}")

with open('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_nDSC_separate_labels.csv', 'w') as file: # TODO 3 tables {5, 7, 9}
    writer = csv.writer(file)
    writer.writerow(metrics)
    writer.writerows(vals)

# '''

''' Plot per list of subjects
df = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5.csv')
ax = sns.boxplot(data=df).set(xlabel="Metric", ylabel="Value")
ax = sns.swarmplot(data=df)
plt.show()
'''

# ''' Plot per metric
# Paths
# df1 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics1.csv')
# df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5.csv')
# df7 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics7.csv')
# df9 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics9.csv')
# Paths seperate labels
# df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_separate_labels.csv')
# df7 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics7_separate_labels.csv')
# df9 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics9_separate_labels.csv')
# Paths nDSC
# df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_nDSC.csv')
# df7 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics7_nDSC.csv')
# df9 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics9_nDSC.csv')
# Paths nDSC separate labels
df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_nDSC_separate_labels.csv')
df7 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics7_nDSC_separate_labels.csv')
df9 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics9_nDSC_separate_labels.csv')
# Dataframes {DSC, Hausdorff, Volume, nDSC}
# data_dsc = [df5['DSC_all'], df7['DSC_all'], df9['DSC_all'], df5['DSC_lens'], df7['DSC_lens'], df9['DSC_lens'], df5['DSC_globe'], df7['DSC_globe'], df9['DSC_globe'],
#             df5['DSC_nerve'], df7['DSC_nerve'], df9['DSC_nerve'], df5['DSC_fats'], df7['DSC_fats'], df9['DSC_fats'], df5['DSC_muscles'], df7['DSC_muscles'], df9['DSC_muscles']]
# data_haus = [df5['Haus_avg_all'], df7['Haus_avg_all'], df9['Haus_avg_all'], df5['Haus_avg_lens'], df7['Haus_avg_lens'], df9['Haus_avg_lens'], df5['Haus_avg_globe'], df7['Haus_avg_globe'],
#             df9['Haus_avg_globe'], df5['Haus_avg_nerve'], df7['Haus_avg_nerve'], df9['Haus_avg_nerve'], df5['Haus_avg_fats'], df7['Haus_avg_fats'], df9['Haus_avg_fats'], 
#             df5['Haus_avg_muscles'], df7['Haus_avg_muscles'], df9['Haus_avg_muscles']]
# data_vol = [df5['Volume_all'], df7['Volume_all'], df9['Volume_all'], df5['Volume_lens'], df7['Volume_lens'], df9['Volume_lens'], df5['Volume_globe'], df7['Volume_globe'], df9['Volume_globe'],
#             df5['Volume_nerve'], df7['Volume_nerve'], df9['Volume_nerve'], df5['Volume_fats'], df7['Volume_fats'], df9['Volume_fats'], df5['Volume_muscles'], df7['Volume_muscles'], df9['Volume_muscles']]
# data_ndsc = [df5['nDSC_all'], df7['nDSC_all'], df9['nDSC_all'], df5['nDSC_lens'], df7['nDSC_lens'], df9['nDSC_lens'], df5['nDSC_globe'], df7['nDSC_globe'], df9['nDSC_globe'],
#             df5['nDSC_nerve'], df7['nDSC_nerve'], df9['nDSC_nerve'], df5['nDSC_fats'], df7['nDSC_fats'], df9['nDSC_fats'], df5['nDSC_muscles'], df7['nDSC_muscles'], df9['nDSC_muscles']]
# Dataframes plus N=1
# data_dsc = [df1['DSC_all'], df5['DSC_all'], df7['DSC_all'], df9['DSC_all'], df1['DSC_lens'], df5['DSC_lens'], df7['DSC_lens'], df9['DSC_lens'], df1['DSC_globe'],df5['DSC_globe'], df7['DSC_globe'], df9['DSC_globe'],
#             df1['DSC_nerve'], df5['DSC_nerve'], df7['DSC_nerve'], df9['DSC_nerve'], df1['DSC_fats'], df5['DSC_fats'], df7['DSC_fats'], df9['DSC_fats'], df1['DSC_muscles'], df5['DSC_muscles'], df7['DSC_muscles'], df9['DSC_muscles']]
# data_haus = [df1['Haus_avg_all'], df5['Haus_avg_all'], df7['Haus_avg_all'], df9['Haus_avg_all'], df1['Haus_avg_lens'], df5['Haus_avg_lens'], df7['Haus_avg_lens'], df9['Haus_avg_lens'], df1['Haus_avg_globe'], df5['Haus_avg_globe'], df7['Haus_avg_globe'],
#             df9['Haus_avg_globe'], df1['Haus_avg_nerve'], df5['Haus_avg_nerve'], df7['Haus_avg_nerve'], df9['Haus_avg_nerve'], df1['Haus_avg_fats'], df5['Haus_avg_fats'], df7['Haus_avg_fats'], df9['Haus_avg_fats'], 
#             df1['Haus_avg_muscles'], df5['Haus_avg_muscles'], df7['Haus_avg_muscles'], df9['Haus_avg_muscles']]
# data_vol = [df1['Volume_all'], df5['Volume_all'], df7['Volume_all'], df9['Volume_all'], df1['Volume_lens'], df5['Volume_lens'], df7['Volume_lens'], df9['Volume_lens'], df1['Volume_globe'], df5['Volume_globe'], df7['Volume_globe'], df9['Volume_globe'],
#             df1['Volume_nerve'], df5['Volume_nerve'], df7['Volume_nerve'], df9['Volume_nerve'], df1['Volume_fats'], df5['Volume_fats'], df7['Volume_fats'], df9['Volume_fats'], df1['Volume_muscles'], df5['Volume_muscles'], df7['Volume_muscles'], df9['Volume_muscles']]
# Dataframes {DSC, Hausdorff, Volume} separate labels
# data_dsc = [df5['DSC_all'], df7['DSC_all'], df9['DSC_all'], df5['DSC_lens'], df7['DSC_lens'], df9['DSC_lens'], df5['DSC_globe'], df7['DSC_globe'], df9['DSC_globe'],
#             df5['DSC_nerve'], df7['DSC_nerve'], df9['DSC_nerve'], df5['DSC_int_fat'], df7['DSC_int_fat'], df9['DSC_int_fat'], df5['DSC_ext_fat'], df7['DSC_ext_fat'], df9['DSC_ext_fat'],
#             df5['DSC_lat_mus'], df7['DSC_lat_mus'], df9['DSC_lat_mus'], df5['DSC_med_mus'], df7['DSC_med_mus'], df9['DSC_med_mus'], df5['DSC_inf_mus'], df7['DSC_inf_mus'], df9['DSC_inf_mus'],
#             df5['DSC_sup_mus'], df7['DSC_sup_mus'], df9['DSC_sup_mus']]
# data_haus = [df5['Haus_avg_all'], df7['Haus_avg_all'], df9['Haus_avg_all'], df5['Haus_avg_lens'], df7['Haus_avg_lens'], df9['Haus_avg_lens'], df5['Haus_avg_globe'], df7['Haus_avg_globe'],
#             df9['Haus_avg_globe'], df5['Haus_avg_nerve'], df7['Haus_avg_nerve'], df9['Haus_avg_nerve'], df5['Haus_avg_int_fat'], df7['Haus_avg_int_fat'], df9['Haus_avg_int_fat'], 
#             df5['Haus_avg_ext_fat'], df7['Haus_avg_ext_fat'], df9['Haus_avg_ext_fat'], df5['Haus_avg_lat_mus'], df7['Haus_avg_lat_mus'], df9['Haus_avg_lat_mus'], df5['Haus_avg_med_mus'], 
#             df7['Haus_avg_med_mus'], df9['Haus_avg_med_mus'], df5['Haus_avg_inf_mus'], df7['Haus_avg_inf_mus'], df9['Haus_avg_inf_mus'], df5['Haus_avg_sup_mus'], df7['Haus_avg_sup_mus'], df9['Haus_avg_sup_mus']]
# data_vol = [df5['Volume_all'], df7['Volume_all'], df9['Volume_all'], df5['Volume_lens'], df7['Volume_lens'], df9['Volume_lens'], df5['Volume_globe'], df7['Volume_globe'], df9['Volume_globe'],
#             df5['Volume_nerve'], df7['Volume_nerve'], df9['Volume_nerve'], df5['Volume_int_fat'], df7['Volume_int_fat'], df9['Volume_int_fat'], df5['Volume_ext_fat'], df7['Volume_ext_fat'], df9['Volume_ext_fat'],
#             df5['Volume_lat_mus'], df7['Volume_lat_mus'], df9['Volume_lat_mus'], df5['Volume_med_mus'], df7['Volume_med_mus'], df9['Volume_med_mus'],df5['Volume_inf_mus'], df7['Volume_inf_mus'], df9['Volume_inf_mus'],
#             df5['Volume_sup_mus'], df7['Volume_sup_mus'], df9['Volume_sup_mus']]
# Dataframes {DSC, Hausdorff, Volume, nDSC} separate labels
data_dsc = [df5['DSC_all'], df7['DSC_all'], df9['DSC_all'], df5['DSC_lens'], df7['DSC_lens'], df9['DSC_lens'], df5['DSC_globe'], df7['DSC_globe'], df9['DSC_globe'],
            df5['DSC_nerve'], df7['DSC_nerve'], df9['DSC_nerve'], df5['DSC_int_fat'], df7['DSC_int_fat'], df9['DSC_int_fat'], df5['DSC_ext_fat'], df7['DSC_ext_fat'], df9['DSC_ext_fat'],
            df5['DSC_lat_mus'], df7['DSC_lat_mus'], df9['DSC_lat_mus'], df5['DSC_med_mus'], df7['DSC_med_mus'], df9['DSC_med_mus'], df5['DSC_inf_mus'], df7['DSC_inf_mus'], df9['DSC_inf_mus'],
            df5['DSC_sup_mus'], df7['DSC_sup_mus'], df9['DSC_sup_mus']]
data_haus = [df5['Haus_avg_all'], df7['Haus_avg_all'], df9['Haus_avg_all'], df5['Haus_avg_lens'], df7['Haus_avg_lens'], df9['Haus_avg_lens'], df5['Haus_avg_globe'], df7['Haus_avg_globe'],
            df9['Haus_avg_globe'], df5['Haus_avg_nerve'], df7['Haus_avg_nerve'], df9['Haus_avg_nerve'], df5['Haus_avg_int_fat'], df7['Haus_avg_int_fat'], df9['Haus_avg_int_fat'], 
            df5['Haus_avg_ext_fat'], df7['Haus_avg_ext_fat'], df9['Haus_avg_ext_fat'], df5['Haus_avg_lat_mus'], df7['Haus_avg_lat_mus'], df9['Haus_avg_lat_mus'], df5['Haus_avg_med_mus'], 
            df7['Haus_avg_med_mus'], df9['Haus_avg_med_mus'], df5['Haus_avg_inf_mus'], df7['Haus_avg_inf_mus'], df9['Haus_avg_inf_mus'], df5['Haus_avg_sup_mus'], df7['Haus_avg_sup_mus'], df9['Haus_avg_sup_mus']]
data_vol = [df5['Volume_all'], df7['Volume_all'], df9['Volume_all'], df5['Volume_lens'], df7['Volume_lens'], df9['Volume_lens'], df5['Volume_globe'], df7['Volume_globe'], df9['Volume_globe'],
            df5['Volume_nerve'], df7['Volume_nerve'], df9['Volume_nerve'], df5['Volume_int_fat'], df7['Volume_int_fat'], df9['Volume_int_fat'], df5['Volume_ext_fat'], df7['Volume_ext_fat'], df9['Volume_ext_fat'],
            df5['Volume_lat_mus'], df7['Volume_lat_mus'], df9['Volume_lat_mus'], df5['Volume_med_mus'], df7['Volume_med_mus'], df9['Volume_med_mus'],df5['Volume_inf_mus'], df7['Volume_inf_mus'], df9['Volume_inf_mus'],
            df5['Volume_sup_mus'], df7['Volume_sup_mus'], df9['Volume_sup_mus']]
data_ndsc = [df5['nDSC_all'], df7['nDSC_all'], df9['nDSC_all'], df5['nDSC_lens'], df7['nDSC_lens'], df9['nDSC_lens'], df5['nDSC_globe'], df7['nDSC_globe'], df9['nDSC_globe'],
            df5['nDSC_nerve'], df7['nDSC_nerve'], df9['nDSC_nerve'], df5['nDSC_int_fat'], df7['nDSC_int_fat'], df9['nDSC_int_fat'], df5['nDSC_ext_fat'], df7['nDSC_ext_fat'], df9['nDSC_ext_fat'],
            df5['nDSC_lat_mus'], df7['nDSC_lat_mus'], df9['nDSC_lat_mus'], df5['nDSC_med_mus'], df7['nDSC_med_mus'], df9['nDSC_med_mus'], df5['nDSC_inf_mus'], df7['nDSC_inf_mus'], df9['nDSC_inf_mus'],
            df5['nDSC_sup_mus'], df7['nDSC_sup_mus'], df9['nDSC_sup_mus']]

# Color palette
# palette = ['blue','orange','green','blue','orange','green','blue','orange','green','blue','orange','green','blue','orange','green', 'blue','orange','green']
# Color palette N=1
# palette = ['red','blue','orange','green','red','blue','orange','green','red','blue','orange','green','red','blue','orange','green','red','blue','orange','green','red','blue','orange','green']
# Color palette separate lables
palette = ['blue','orange','green','blue','orange','green','blue','orange','green','blue','orange','green','blue','orange','green','blue','orange','green', 'blue','orange','green','blue','orange','green',
            'blue','orange','green', 'blue','orange','green']
# Subplots
fig, axs = plt.subplots(4, sharex=True)
fig.suptitle('Similarity metrics')
# Single plot
# ax1 = sns.boxplot(data=data_dsc, palette=palette).set(xlabel="Label", ylabel="Value")
# ax1 = sns.swarmplot(data=data_dsc, palette=palette)
# ax2 = sns.boxplot(data=data_haus, palette=palette).set(xlabel="Label", ylabel="Value")
# ax2 = sns.swarmplot(data=data_haus, palette=palette)
# ax3 = sns.boxplot(data=data_vol, palette=palette).set(xlabel="Label", ylabel="Value")
# ax3 = sns.swarmplot(data=data_vol, palette=palette)
# Boxplot & Swarmplot (points)
ax1 = sns.boxplot(data=data_dsc, palette=palette, ax=axs[0]).set(ylabel="Value")
ax1 = sns.swarmplot(data=data_dsc, palette=palette, ax=axs[0])
ax2 = sns.boxplot(data=data_haus, palette=palette, ax=axs[1]).set(ylabel="Value")
ax2 = sns.swarmplot(data=data_haus, palette=palette, ax=axs[1])
ax3 = sns.boxplot(data=data_vol, palette=palette, ax=axs[2]).set(ylabel="Value")
ax3 = sns.swarmplot(data=data_vol, palette=palette, ax=axs[2])
ax4 = sns.boxplot(data=data_ndsc, palette=palette, ax=axs[3]).set(xlabel="Label", ylabel="Value")
ax4 = sns.swarmplot(data=data_ndsc, palette=palette, ax=axs[3])
# Legend
legend_elements = [
    Line2D([0], [0], color='blue', lw=4, label='N=5'),
    Line2D([0], [0], color='orange', lw=4, label='N=7'),
    Line2D([0], [0], color='green', lw=4, label='N=9')
]
# Legend N=1
# legend_elements = [Line2D([0], [0], color='red', lw=4, label='N=1'),
#     Line2D([0], [0], color='blue', lw=4, label='N=5'),
#     Line2D([0], [0], color='orange', lw=4, label='N=7'),
#     Line2D([0], [0], color='green', lw=4, label='N=9')]
ax1.legend(handles=legend_elements)
ax2.legend(handles=legend_elements)
ax3.legend(handles=legend_elements)
ax4.legend(handles=legend_elements)
# Labels and title
# ax1.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','fats','fats','fats','muscles','muscles','muscles'])
# Labels and title separate labels
ax1.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','int_fat','int_fat','int_fat','ext_fat','ext_fat','ext_fat',
                    'lat_mus','lat_mus','lat_mus','med_mus','med_mus','med_mus','inf_mus','inf_mus','inf_mus','sup_mus','sup_mus','sup_mus'])
# Labels and title N=1
# ax1.set_xticklabels(['all','all','all','all','lens','lens','lens','lens','globe','globe','globe','globe','nerve','nerve','nerve','nerve','fats','fats','fats','fats','muscles','muscles','muscles','muscles'])
ax1.set_title('DSC')
# Labels and title
# ax2.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','fats','fats','fats','muscles','muscles','muscles'])
# Labels and title separate labels
ax2.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','int_fat','int_fat','int_fat','ext_fat','ext_fat','ext_fat',
                    'lat_mus','lat_mus','lat_mus','med_mus','med_mus','med_mus','inf_mus','inf_mus','inf_mus','sup_mus','sup_mus','sup_mus'])
# Labels and title N=1
# ax2.set_xticklabels(['all','all','all','all','lens','lens','lens','lens','globe','globe','globe','globe','nerve','nerve','nerve','nerve','fats','fats','fats','fats','muscles','muscles','muscles','muscles'])
ax2.set_title('Hausdorff distance')
# Labels and title 
# ax3.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','fats','fats','fats','muscles','muscles','muscles'])
# Labels and title separate labels
ax3.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','int_fat','int_fat','int_fat','ext_fat','ext_fat','ext_fat',
                    'lat_mus','lat_mus','lat_mus','med_mus','med_mus','med_mus','inf_mus','inf_mus','inf_mus','sup_mus','sup_mus','sup_mus'])
# Labels and title N=1
# ax3.set_xticklabels(['all','all','all','all','lens','lens','lens','lens','globe','globe','globe','globe','nerve','nerve','nerve','nerve','fats','fats','fats','fats','muscles','muscles','muscles','muscles'])
ax3.set_title('Volume similarity')
# Labels and title
# ax4.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','fats','fats','fats','muscles','muscles','muscles'])
# Labels and title separate labels
ax4.set_xticklabels(['all','all','all','lens','lens','lens','globe','globe','globe','nerve','nerve','nerve','int_fat','int_fat','int_fat','ext_fat','ext_fat','ext_fat',
                    'lat_mus','lat_mus','lat_mus','med_mus','med_mus','med_mus','inf_mus','inf_mus','inf_mus','sup_mus','sup_mus','sup_mus'])
ax4.set_title('nDSC')

# Save
# plt.savefig('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/dsc.png')
# Show
plt.show()
# '''