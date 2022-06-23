import SimpleITK as sitk

gt_path = 'path/to/groundtruth.nii.gz'
pr_path = 'path/to/preditction.nii.gz'

reader = sitk.ImageFileReader()

reader.SetFileName(gt_path)
gt_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

reader.SetFileName(pr_path)
pr_sitk = sitk.Cast(reader.Execute(), sitk.sitkUInt8)

overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
overlap_measures_filter.Execute(gt_sitk, pr_sitk)

dsc = overlap_measures_filter.GetDiceCoefficient()

# sitk.HausdorffDistanceImageFilter()