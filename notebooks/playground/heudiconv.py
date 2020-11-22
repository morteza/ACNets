# %%
from pathlib import Path
import pandas as pd
import numpy as np

import os
import subprocess

print('working directory:', os.getcwd())


raw_data_path = Path('data/julia2018_raw/RawDicom_A1_A2/Attention')

jsubs = [p.stem for p in raw_data_path.glob('*VGP*')]
sessions = ['1', '2']


for ses in sessions:
  for jsub in jsubs:
    cmd = ("heudiconv"
           " --bids"
           " -f python/preprocessing/heuristic.py"
           f" --files {raw_data_path}/{jsub}/RawDICOM/Attention{ses}"
           f" -s {jsub}"
           f" -ss {ses}"
           " -o data/julia2018_raw_task"
           " -c dcm2niix"
           " --datalad")
    print(cmd)

    res = subprocess.run(cmd.split(' '), stderr=subprocess.PIPE, text=True, cwd=os.getcwd())

    print(res.stderr)

print('DONE!')
