#%% dicom2nifti
import dicom2nifti
import os

root_dir = '/Users/morteza/Downloads/Julia/Resting_State/Resting_State'
output_dir = '/Users/morteza/Downloads/Julia_resting_nii'
os.makedirs(output_dir,exist_ok=True)

subjects = ['NVGP01','NVGP03']

for subj in subjects:
  dicom_dir = f'{root_dir}/{subj}/{subj}_bold/'
  output_file = f'{output_dir}/{subj}.nii.gz'
  dicom2nifti.dicom_series_to_nifti(dicom_dir, output_file)


#%% dcm2