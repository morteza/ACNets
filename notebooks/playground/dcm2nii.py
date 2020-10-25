#%% dicom2nifti
import dicom2nifti
import os

root_dir = 'data/julia2018_raw/Resting_State/Resting_State'
output_dir = 'data/julia2018_rest/'

os.makedirs(output_dir, exist_ok=True)

subjects = ['NVGP01', 'NVGP03']

for subj in subjects:
  dicom_dir = f'{root_dir}/{subj}/{subj}_bold/'
  output_file = f'{output_dir}/{subj}.nii.gz'
  dicom2nifti.dicom_series_to_nifti(dicom_dir, output_file)

#%% dcm2