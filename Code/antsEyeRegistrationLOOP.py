import os

base_dir = '/mnt/sda1/ANTs/a123/'
ref_mni152 = '/mnt/sda1/ANTs/input/mni152/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'
ref_colin27 = '/mnt/sda1/ANTs/input/colin27/tpl-MNIColin27_T1w.nii.gz'
eye_mask_mni = '/mnt/sda1/ANTs/input/mni152/tpl-MNI152NLin2009cAsym_res-01_desc-eye_mask.nii.gz'

# antsEyeExtraction and antsApplyTransforms
for folder1 in os.listdir(base_dir):
    # print(folder1)
    input_t1 = base_dir + folder1 + '/input/' + folder1 + '_T1.nii.gz'
    input_labels = base_dir + folder1 + '/input/' + folder1 + '_labels.nii.gz'
    input_t1_cropped = base_dir + folder1 + '/input/' + folder1 + '_T1_cropped.nii.gz'
    input_labels_cropped = base_dir + folder1 + '/input/' + folder1 + '_labels_cropped.nii.gz'
    ref_mni152_cropped = base_dir + folder1 + '/input/' + 'tpl-MNI152NLin2009cAsym_res-01_T1w_cropped.nii.gz'
    ref_colin27_cropped = base_dir + folder1 + '/input/' + 'tpl-MNIColin27_T1w_cropped.nii.gz'
    output = base_dir + folder1 + '/output_colin27_cropped/' # Change this when doing new extractions
    # os.makedirs(output)

    # Brain extraction (only to test)
    # command1 = 'antsBrainExtraction.sh -d 3' + \
    # ' -a ' + input_t1 + \
    # ' -e ' + ref_mni152 + \
    # ' -m ' + eye_mask_mni + \
    # ' -o ' + output + \
    # ' -k ' + '1' # 1 = keep temporary files, 0 = remove them
    # print(command1)
    # # os.system(command1)

    # Eye extraction
    # command1 = 'antsEyeExtraction.sh -d 3' + \
    # ' -a ' + input_t1     + \
    # ' -e ' + ref_colin27  + \
    # ' -f ' + eye_mask_mni + \
    # ' -g ' + input_labels + \
    # ' -o ' + output       + \
    # ' -k ' + '1' # 1 = keep temporary files, 0 = remove them
    # print(command1)
    # os.system(command1)

    # ApplyTransforms
    # command2 = 'antsApplyTransforms -d 3 '                                + \
    # ' -i ' +  input_labels                                                + \
    # ' -o ' +  output + 'all_segments_template.nii.gz'                     + \
    # ' -r ' +  ref_colin27                                                 + \
    # ' -n ' + 'MultiLabel' + \
    # ' -t ' + '[' + output + 'BrainExtractionPrior0GenericAffine.mat, 0 ]' + \
    # ' -t ' + output + 'BrainExtractionPrior1Warp.nii.gz'                  + \
    # ' --float 0 --verbose 1'
    # print(command2)
    # os.system(command2)

    # antsRegistrationSyN (for cropped images)
    command1 = 'antsRegistrationSyN.sh -d 3' + \
    ' -m ' + input_t1_cropped   + \
    ' -f ' + ref_colin27_cropped + \
    ' -t ' + 's'                + \
    ' -o ' + output
    print(command1)
    os.system(command1)

    # ApplyTransforms (for cropped images)
    command2 = 'antsApplyTransforms -d 3 ' + \
    ' -i ' +  input_labels_cropped + \
    ' -o ' +  output + 'all_segments_template.nii.gz' + \
    ' -r ' +  ref_colin27_cropped + \
    ' -n ' + 'MultiLabel' + \
    ' -t ' + '[' + output + '0GenericAffine.mat, 0 ]' + \
    ' -t ' + output + '1Warp.nii.gz' + \
    ' --float 0 --verbose 1'
    print(command2)
    os.system(command2)