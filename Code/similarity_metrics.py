import SimpleITK as sitk

gt_path = '/mnt/sda1/Repos/a-eye/Data/templateflow/colin27/tpl-MNIColin27_T1w.nii.gz'
pr_path = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/sub-01/output_colin27/BrainExtractionNormalized.nii.gz'

reader = sitk.ImageFileReader()

reader.SetFileName(gt_path)
gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

reader.SetFileName(pr_path)
pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
overlap_measures_filter.Execute(gt_sitk, pr_sitk)

dsc = overlap_measures_filter.GetDiceCoefficient()
print(dsc)

hausdorf = sitk.HausdorffDistanceImageFilter()
hausdorf.Execute(gt_sitk, pr_sitk)
hausdorf_distance = hausdorf.GetHausdorffDistance()
print(hausdorf_distance)
hausdorf_distance_avg = hausdorf.GetAverageHausdorffDistance()
print(hausdorf_distance_avg)

# Dice
# dice_txt = output_dir + 'Dice.txt'
# arr_dice = genfromtxt(dice_txt)
# # print(arr_demons)
# arr5_dice = np.sort(arr_dice)[::-1][:5] # 5th higher values
# print(f'Dice: {arr5_dice}')
# arr_ind = np.argsort(arr_dice)[::-1][:5] # Indexes of first 5 higher values
# # print(arr_ind)
# arr5_sub = arr_sub[arr_ind]
# print(f'Dice: {arr5_sub}')

# Hausdorff 
# haus_txt = output_dir + 'Hausdorff_avg.txt'
# arr_haus = genfromtxt(haus_txt)
# # print(arr_demons)
# arr5_haus = np.sort(arr_haus)[::-1][:5] # 5th higher values
# print(f'Hausdorff: {arr5_haus}')
# arr_ind = np.argsort(arr_haus)[::-1][:5] # Indexes of first 5 higher values
# # print(arr_ind)
# arr5_sub = arr_sub[arr_ind]
# print(f'Hausdorff: {arr5_sub}')