import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 24
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['axes.labelsize'] = 18

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

uv_path = Path.joinpath(INIT_DATA_2025).parent
print(uv_path)
uv_spectra = ut.load_spectra_data(Path.joinpath(uv_path, "UV","absorption"), uv.read_csv_uv)

fig,ax = plt.subplots(1,1,dpi=200,figsize=(8,6))

i=0

for uv_df in uv_spectra:
    if uv_df.loc[600:700].mean()['absorption'] > 0.05:
        continue
    if '2025' in uv_df.attrs['name'] and uv_df.attrs['class'] == 'ADOM':
        
        uv.plot_uv(uv_df,ax=ax)
    
    if i > 9:
        break
        
    i += 1
       
plt.title('')        
plt.xlim(250,450)
plt.ylim(0,1)
plt.tight_layout()        
plt.show()
