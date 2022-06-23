import nibabel as nb
import numpy as np
from pathlib import Path

base_dir = '/mnt/sda1/ANTs/a123/'
output_dir = '/mnt/sda1/ANTs/'

# Create mask from all subjects' labels
segments = [nb.load(f) for f in Path(base_dir).glob("*/output_colin27/all_segments_template.nii.gz")]
# print(segments[0])
segmentation = np.zeros_like(segments[0].dataobj, dtype="uint8")
for label, volume in enumerate(segments):
    segmentation[np.asanyarray(volume.dataobj) > 0] = 1
header = segments[0].header.copy()
header.set_data_dtype("uint8")
nii = nb.Nifti1Image(segmentation, segments[0].affine, header)
# print(nii.header)
nii.to_filename(output_dir+"all_segments_mask.nii.gz")