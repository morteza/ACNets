# %%
from bids import BIDSLayout
from pathlib import Path

project_dir = Path('.')

layout = BIDSLayout(project_dir / 'data/julia2018',
                    derivatives=True,
                    database_path=project_dir / 'tmp/pybids_cache/julia2018')


# layout.add_derivatives('fmriprep')



task = 'attention'
sessions = ['1','2']
subjects = ['NVGP01']
# subjects = layout.get_subject()


fmri_images = layout.get(task=task,
                         desc='preproc',
                         subject=subjects,
                         session=sessions,
                         suffix='bold',
                         scope='fmriprep',
                         extension='nii.gz',
                         return_type='filename')

mask_images = layout.get(task=task,
                         desc='brain',
                         subject=subjects,
                         session=sessions,
                         suffix='mask',
                         scope='fmriprep',
                         extension='nii.gz',
                         return_type='filename')

confounds_files = layout.get(task=task,
                             desc='confounds',
                             subject=subjects,
                             session=sessions,
                             suffix='timeseries',
                             scope='fmriprep',
                             extension='tsv',
                             return_type='filename')

events_files = layout.get(task=task,
                          desc='preproc',
                          subject=subjects,
                          session=sessions,
                          suffix='events',
                          scope='fmriprep',
                          extension='tsv',
                          return_type='filename')


confounds_cols = ['trans_x', 'trans_y', 'trans_z',
                  'rot_x', 'rot_y', 'rot_z',
                  'global_signal',
                  'a_comp_cor_00', 'a_comp_cor_01']

TR = layout.get_tr(task=task,subject=subjects, session=sessions)
