import numpy as np
import nibabel as nib
import SimpleITK as sitk

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/best_subjects_eye_cc/CustomTemplate_5_n1/' # {1, 5, 7, 9}
gt_path = base_dir + 'Probability_Maps/prob_map_cropped_th0.nii.gz' # GT
pr_path = base_dir + 'reg_cropped_other_subjects/sub-01_reg_cropped/labels2template.nii.gz' # Labels' image to compare to GT

# forma 1 --> works
# gt_im = nib.load(gt_path)
# pr_im = nib.load(pr_path)
# gt_arr = np.array(gt_im.dataobj)
# pr_arr = np.array(pr_im.dataobj)

# forma 2 --> doesn't work --> do GetArrayFromImage() instead
reader = sitk.ImageFileReader()
reader.SetFileName(gt_path)
gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)
reader.SetFileName(pr_path)
pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

# forma 3 --> funciona
gt_arr = sitk.GetArrayFromImage(gt_sitk)
pr_arr = sitk.GetArrayFromImage(pr_sitk)

print(f'gt_arr: {gt_arr.shape}, pr_arr: {pr_arr.shape}')
print(f'gt: {np.count_nonzero(gt_arr==1)}, pr: {np.count_nonzero(pr_arr==1)}')

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
            print(f'np.sum(gt)={np.sum(gt)}, len(gt.flatten())={len(gt.flatten())}')
        tp = np.sum(seg[gt == 1])
        print(f'tp: {tp}')
        fp = np.sum(seg[gt == 0])
        print(f'fp: {fp}')
        fn = np.sum(gt[seg == 0])
        print(f'fn: {fn}')
        fp_scaled = k * fp
        print(f'fp_scaled: {fp_scaled}')
        dsc_norm = 2 * tp / (fp_scaled + 2 * tp + fn)

        dsc = 2 * tp / (fp + 2 * tp + fn)

        fpr = fp / (len(gt.flatten()) - np.sum(gt))
        if np.sum(gt) == 0:
            fnr = 1.0
        else:
            fnr = fn / np.sum(gt)
        return dsc_norm, dsc # fpr, fnr

gt_mask = gt_arr != 0
pr_mask = pr_arr != 0
# print(f'gt_mask: {gt_mask.shape}, pr_mask: {pr_mask.shape}')

# gt_mask = gt_arr==4 or gt_arr==5
# pr_mask = pr_arr==4 or gt_arr==5

print(f'ndsc vs dsc: {dice_norm_metric( gt_arr==1, pr_arr==1 )}')