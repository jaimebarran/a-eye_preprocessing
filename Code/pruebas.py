import ants
import os
import glob
import numpy as np

base_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/ANTs/a123/'
output_dir = '/mnt/sda1/Repos/a-eye/a-eye_preprocessing/Output/CustomTemplate/'

best_subjects_cc = ['sub-29','sub-20','sub-33','sub-03','sub-02']
best_subjects_mi = ['sub-20','sub-33','sub-09','sub-32','sub-29']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

subs = [] # list
for i in range(len(best_subjects_cc)):
    # print(i)
    subs.append(base_dir + best_subjects_cc[i] + '/input/' + best_subjects_cc[i] + '_T1_hdr.nii.gz')
    # print(subs[i])
# print(subs)

population = list()
for i in range(len(subs)):
    population.append(ants.image_read(subs[i], dimension=3))
    print(population[i])