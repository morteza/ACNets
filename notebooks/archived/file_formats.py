#%%

from scipy.io import loadmat

a = loadmat('/Users/morteza/Downloads/Julia/Behavioral_Data/Behavioral_Data/NVGP01/A1/NVGP01_A1.mat')['trial'][0,1]

type(a)


# %%
# ref: https://nipy.org/nibabel/nifti_images.html
# ref: https://youtu.be/9ffUQo2mF6w
import nibabel as nib
from nilearn import image

example_path = '/Users/morteza/Downloads/Julia/Resting_State/Resting_State/NVGP01/NVGP01_bold/NVGP01_resting.nii.gz'


#nibabel: img = nib.load(example_path)

# nilearn
rsn = image.load_img(example_path)
#header = img.header
#print(header)
rsn.shape

#%%
#plot

from nilearn import plotting
first_rsn = image.index_img(rsn, 0)
#plotting.plot_stat_map(first_rsn)

#%%

selected_images = image.index_img(rsn, slice(1,3))

for img in image.iter_img(selected_images):
  plotting.plot_stat_map(img,
                         threshold=390,
                         display_mode='z')




# %%
# Test whether provided and extracted Nifti files are the same.
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt 
from IPython.display import display
import seaborn as sns; sns.set()

root = Path('data/julia2018_raw/Resting_State/Resting_State/NVGP01')

raw_bold = root / 'NVGP01_bold' / 'NVGP01_resting.nii.gz'
raw_magn = root / 'Magnitude_Image' / 'NVGP01_Mag_fieldmap_brain.nii.gz'
raw_phas = root / 'Phase_Image' / 'NVGP01_Phase_rad_s.nii.gz'

bids_root = Path('data/julia2018/sub-NVGP01/ses-rest')
bids_bold = bids_root / 'func' / 'sub-NVGP01_ses-rest_task-rest_bold.nii.gz'

raw_img = nib.load(raw_bold)
raw_img_data = raw_img.get_fdata()

bids_img = nib.load(bids_bold)
bids_img_data = bids_img.get_fdata()

are_bids_raw_equal = (bids_img_data == raw_img_data).all()

print('Are both provided and extracted Nifti files the same?',
      are_bids_raw_equal)

# plt.hist(bids_img_data.flatten())
# plt.show()
# plt.hist(raw_img_data.flatten())
# plt.show()

# plt.imshow(nib.load(raw_phas).get_fdata()[:,:,18])

# plt.hist(nib.load(raw_phas).get_fdata().flatten())

nib.load(raw_magn).get_fdata().min()