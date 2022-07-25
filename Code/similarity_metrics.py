from asyncore import write
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt


# ''' Data frame file generation
base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/' # {5, 7, 9}
gt_path = base_dir + 'Probability_Maps/prob_map_cropped_th0.nii.gz' # GT

reader = sitk.ImageFileReader()
reader.SetFileName(gt_path)
gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

# List of best subjects
best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-33'] # 5
# best_subjects_cc = ['sub-02','sub-03','sub-20','sub-29','sub-30','sub-33','sub-34'] # 7
# best_subjects_cc = ['sub-02','sub-03','sub-08','sub-09','sub-20','sub-29','sub-30','sub-33','sub-34'] # 9

# List of remaining subjects
all_subjects = list()
for i in range(35):
    all_subjects.append('sub-'+str(i+1).zfill(2))
rest_subjects = [elem for elem in all_subjects if elem not in best_subjects_cc]

# Save values in an array
val_dsc = np.zeros(len(rest_subjects))
val_hau = np.zeros(len(rest_subjects))
val_hau_avg = np.zeros(len(rest_subjects))
val_vol = np.zeros(len(rest_subjects))

for i in range(len(rest_subjects)):

    # Prediction image to compare to GT
    pr_path = base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2template.nii.gz' # Labels' image to compare to GT
    reader.SetFileName(pr_path)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

    # Measures Image Filter
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.Execute(gt_sitk==3, pr_sitk==3)

    # DSC
    dsc = overlap_measures_filter.GetDiceCoefficient() # Get the mean overlap (Dice coefficient) over all labels
    val_dsc[i] = dsc

    # Volume
    vol = overlap_measures_filter.GetVolumeSimilarity() # Get the volume similarity over all labels
    val_vol[i] = vol

    # Hausdorff distance
    hausdorf = sitk.HausdorffDistanceImageFilter()
    hausdorf.Execute(gt_sitk==3, pr_sitk==3)
    # hausdorf_distance = hausdorf.GetHausdorffDistance()
    # val_hau[i] = hausdorf_distance
    hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance() # Return the computed Hausdorff distance
    val_hau_avg[i] = hausdorf_distance_avg

# Save values to a csv
# metrics = ['Subject','DSC', 'Hausdorff', 'Hausdorff_avg']
metrics = ['Subject','DSC', 'Hausdorff_avg', 'Volume']
# print(val_dsc[0], val_hau[0], val_hau_avg[0])
vals = np.array([rest_subjects, val_dsc, val_hau_avg, val_vol])
vals = vals.T
# print(vals)
# print(f"type: {vals.dtype}, shape: {vals.shape}")

with open('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5_optic_nerve_3.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(metrics)
    writer.writerows(vals)

# '''

# ''' Plot per list of subjects
df = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5.csv')
ax = sns.boxplot(data=df).set(xlabel="Metric", ylabel="Value")
ax = sns.swarmplot(data=df)
plt.show()
# '''


''' Plot per metric
df5 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics5.csv')
df7 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics7.csv')
df9 = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/metrics9.csv')
data_dsc = [df5['DSC'], df7['DSC'], df9['DSC']]
ax = sns.boxplot(data=data_dsc).set(xlabel="N subjects", ylabel="Value")
ax = sns.swarmplot(data=data_dsc)
plt.show()
# '''