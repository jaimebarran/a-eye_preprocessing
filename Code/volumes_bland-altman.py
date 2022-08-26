from asyncore import write
from turtle import title
import SimpleITK as sitk
import numpy as np
import pandas as pd
import csv
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import Line2D
from sqlalchemy import true
from scipy import stats
import pingouin as pg
import statsmodels.api as sm


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
val_vol_pr_all = np.zeros(len(rest_subjects))
val_vol_gt_all = np.repeat(np.count_nonzero(gt_arr), len(rest_subjects))
# Lens
val_vol_pr_lens = np.zeros(len(rest_subjects))
val_vol_gt_lens = np.repeat(np.count_nonzero(gt_arr==1), len(rest_subjects))
# Globe
val_vol_pr_globe = np.zeros(len(rest_subjects))
val_vol_gt_globe = np.repeat(np.count_nonzero(gt_arr==2), len(rest_subjects))
# Optic nerve
val_vol_pr_nerve = np.zeros(len(rest_subjects))
val_vol_gt_nerve = np.repeat(np.count_nonzero(gt_arr==3), len(rest_subjects))
# Intraconal fat
val_vol_pr_int_fat = np.zeros(len(rest_subjects))
val_vol_gt_int_fat = np.repeat(np.count_nonzero(gt_arr==4), len(rest_subjects))
# Extraconal fat
val_vol_pr_ext_fat = np.zeros(len(rest_subjects))
val_vol_gt_ext_fat = np.repeat(np.count_nonzero(gt_arr==5), len(rest_subjects))
# Lateral rectus muscle
val_vol_pr_lat_mus = np.zeros(len(rest_subjects))
val_vol_gt_lat_mus = np.repeat(np.count_nonzero(gt_arr==6), len(rest_subjects))
# Medial rectus muscle
val_vol_pr_med_mus = np.zeros(len(rest_subjects))
val_vol_gt_med_mus = np.repeat(np.count_nonzero(gt_arr==7), len(rest_subjects))
# Inferior rectus muscle
val_vol_pr_inf_mus = np.zeros(len(rest_subjects))
val_vol_gt_inf_mus = np.repeat(np.count_nonzero(gt_arr==8), len(rest_subjects))
# Superior rectus muscle
val_vol_pr_sup_mus = np.zeros(len(rest_subjects))
val_vol_gt_sup_mus = np.repeat(np.count_nonzero(gt_arr==9), len(rest_subjects))

for i in range(len(rest_subjects)):

    # Prediction image to compare to GT
    pr_path = base_dir + 'reg_cropped_other_subjects/' + rest_subjects[i] + '_reg_cropped/labels2template.nii.gz' # Labels' image to compare to GT
    reader.SetFileName(pr_path)
    pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
    pr_arr = sitk.GetArrayFromImage(pr_sitk) # in numpy format
    # pr_size = pr_arr.shape[0]*pr_arr.shape[1]*pr_arr.shape[2]

    # LENS
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==1)
    val_vol_pr_lens[i] = vol_pr

    # GLOBE EX LENS
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==2)
    val_vol_pr_globe[i] = vol_pr

    # OPTIC NERVE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==3)
    val_vol_pr_nerve[i] = vol_pr

    # INTRACONAL FAT
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==4)
    val_vol_pr_int_fat[i] = vol_pr

    # EXTRACONAL FAT
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==5)
    val_vol_pr_ext_fat[i] = vol_pr

    # LATERAL RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==6)
    val_vol_pr_lat_mus[i] = vol_pr

    # MEDIAL RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==7)
    val_vol_pr_med_mus[i] = vol_pr

    # INFERIOR RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==8)
    val_vol_pr_inf_mus[i] = vol_pr

    # SUPERIOR RECTUS MUSCLE
    # Volume prediction
    vol_pr = np.count_nonzero(pr_arr==9)
    val_vol_pr_sup_mus[i] = vol_pr

    # ALL LABELS
    # Volume structure
    vol_pr = np.count_nonzero(pr_arr)
    val_vol_pr_all[i] = vol_pr


# Save values to a csv
metrics = ['Subject','vol_pr_all','vol_gt_all','vol_pr_lens','vol_gt_lens','vol_pr_globe','vol_gt_globe','vol_pr_nerve','vol_gt_nerve',
            'vol_pr_int_fat','vol_gt_int_fat','vol_pr_ext_fat','vol_gt_ext_fat','vol_pr_lat_mus','vol_gt_lat_mus','vol_pr_med_mus','vol_gt_med_mus',
            'vol_pr_inf_mus','vol_gt_inf_mus','vol_pr_sup_mus','vol_gt_sup_mus']
vals = np.array([rest_subjects, val_vol_pr_all, val_vol_gt_all, val_vol_pr_lens, val_vol_gt_lens, val_vol_pr_globe, val_vol_gt_globe, val_vol_pr_nerve,
                val_vol_gt_nerve,val_vol_pr_int_fat, val_vol_gt_int_fat, val_vol_pr_ext_fat, val_vol_gt_ext_fat, val_vol_pr_lat_mus, val_vol_gt_lat_mus,
                val_vol_pr_med_mus, val_vol_gt_med_mus, val_vol_pr_inf_mus, val_vol_gt_inf_mus, val_vol_pr_sup_mus, val_vol_gt_sup_mus])
vals = vals.T

with open('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/volumes_bland-altman.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(metrics)
    writer.writerows(vals)

# '''

# ''' Bland-Altman plot
df_vol = pd.read_csv('/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/volumes_bland-altman.csv')

fig, ax = plt.subplots(2, 5, figsize=(20,10), sharex=True, sharey=True)
fig.canvas.set_window_title('Volume difference - Bland-Altman plots')
# fig.suptitle('Volume difference')

# all labels
sm.graphics.mean_diff_plot(df_vol['vol_gt_all'], df_vol['vol_pr_all'], ax=ax[0][0])
ax[0][0].set_title('all')
# lens
sm.graphics.mean_diff_plot(df_vol['vol_gt_lens'], df_vol['vol_pr_lens'], ax=ax[0][1])
ax[0][1].set_title('lens')
# globe
sm.graphics.mean_diff_plot(df_vol['vol_gt_globe'], df_vol['vol_pr_globe'], ax=ax[0][2])
ax[0][2].set_title('globe')
# nerve
sm.graphics.mean_diff_plot(df_vol['vol_gt_nerve'], df_vol['vol_pr_nerve'], ax=ax[0][3])
ax[0][3].set_title('nerve')
# int fat
sm.graphics.mean_diff_plot(df_vol['vol_gt_int_fat'], df_vol['vol_pr_int_fat'], ax=ax[0][4])
ax[0][4].set_title('int fat')
# ext fat
sm.graphics.mean_diff_plot(df_vol['vol_gt_ext_fat'], df_vol['vol_pr_ext_fat'], ax=ax[1][0])
ax[1][0].set_title('ext fat')
# lat mus
sm.graphics.mean_diff_plot(df_vol['vol_gt_lat_mus'], df_vol['vol_pr_lat_mus'], ax=ax[1][1])
ax[1][1].set_title('lat mus')
# med mus
sm.graphics.mean_diff_plot(df_vol['vol_gt_med_mus'], df_vol['vol_pr_med_mus'], ax=ax[1][2])
ax[1][2].set_title('med mus')
# inf mus
sm.graphics.mean_diff_plot(df_vol['vol_gt_inf_mus'], df_vol['vol_pr_inf_mus'], ax=ax[1][3])
ax[1][3].set_title('inf mus')
# sup mus
sm.graphics.mean_diff_plot(df_vol['vol_gt_sup_mus'], df_vol['vol_pr_sup_mus'], ax=ax[1][4])
ax[1][4].set_title('sup mus')

plt.tight_layout()
plt.show()
# '''