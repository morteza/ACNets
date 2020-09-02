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
