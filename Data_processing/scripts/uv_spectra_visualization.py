import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams["legend.fontsize"] = 17

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import INIT_DATA, CACHE_DATA_PATH, SPECTRA_PATH

uv_path = Path.joinpath(INIT_DATA)
fluo_path = Path.joinpath(INIT_DATA, "Fluorescence")
fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)

print(len(fluo_spectra))
uv_spectra = ut.load_spectra_data(Path.joinpath(uv_path, "UV","absorption"), uv.read_csv_uv)

fig,ax = plt.subplots(1,1,dpi=300,figsize=(10,8))

i=0

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
sample_list = collection.samples


name_list = ['ADOM-SL1-7-2025','ADOM-SL2-8-2025','ADOM-SL3-8-2025','ADOM-SL4-2-2025','ADOM-SL5-2-2025'
             ,'ADOM-SL6-9-2025','ADOM-SL7-2-2025','ADOM-SL9-1-2025','ADOM-SL10-1-2025','ADOM-SLB-B-1-2025']
name_list = ['LG-11-2-30-2024','LST-S-SBP-2024','PHA-T598-2024','SFA-PW04-2024','CHA-GL02-2024']

for sample_name in sample_list.keys():
    

    if sample_name in name_list:
        
        name = sample_name
        short_name = name[:-5]
        
        if name == 'ADOM-SLB-B-1-2025':
            short_name = 'ADOM-B'
        print(sample_name)
        uv_df = sample_list[sample_name].uv_vis_absorption
        if uv_df is None:
            raise ValueError()    
        uv_df.attrs['name'] = short_name
        uv_df.attrs['TOC'] = sample_list[sample_name].org_carbon
        uv.plot_uv(uv_df,ax=ax,norm_by_TOC=True)
        
    i += 1
       
plt.title('')        
plt.xlim(210,800)
plt.ylim(0,0.1)
plt.legend()
plt.tight_layout()        
plt.savefig(r"C:\Users\Kirill\Documents\Figure_1.png")

fig,ax = plt.subplots(1,1,dpi=300,figsize=(10,8))

for sample_name in sample_list.keys():
    

    if sample_name in name_list:
        
        name = sample_name
        short_name = name[:-5]
        
        if name == 'ADOM-SLB-B-1-2025':
            short_name = 'ADOM-B'
        
        fluo = sample_list[sample_name].fluorescence_eem
        
        if fluo is None:
            raise ValueError()

        fluo.attrs['name'] = short_name
        
        fl.plot_2d(fluo,ex_wave=280,ax=ax,norm=True,raman_spline=True)
        
plt.title('')
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')
plt.tight_layout()         
plt.savefig(r"C:\Users\Kirill\Documents\Figure_2.png")            
