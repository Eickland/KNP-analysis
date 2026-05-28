import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl
import warnings

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams["legend.fontsize"] = 12

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Production.statistic import box_plot_default,swarm_plot_default,strip_plot_default
from Data_processing.Utilits.config_paths import INIT_DATA, CACHE_DATA_PATH, STAT_PATH,GRAPH_PATH

data = pd.read_csv(Path.joinpath(STAT_PATH,"all_descriptors.csv"))
data.dropna(inplace=True)
    
print(data)
print(data.columns)
data = data[data['Subclass'] != 'Storage 8']
data = data[data['sample_id'].str.contains('2025',na=False)]
print(data.shape)
print(data['Subclass'].value_counts())
data.drop(columns=['lambda'], inplace=True)
#data = ut.filter_by_iqr_per_group(data,iqr_param=1.5,target_column='Subclass', filter_columns=["Ag_380","org_carbon","Ag_275"])
data = data[data['Ag_275'] < 140]
data = data[data['Ag_380'] < 30]
data = data[data['Asm_280'] < 2.5]
data = data[data['Asm_350'] < 6.5]
data = data[data['E3E4'] < 700]
data = data[data['E4E6'] < 300]
data = data[data['Suva'] < 0.055]
print(data.shape)

#strip_plot_default(data,path=str(Path.joinpath(GRAPH_PATH,'stripplot')), save=True,target_column='Subclass')
#swarm_plot_default(data,path=str(Path.joinpath(GRAPH_PATH,'swarmplot')), save=True,target_column='Subclass')
box_plot_default(data,path=str(Path.joinpath(GRAPH_PATH,'boxplot')), save=True,target_column='Subclass')
data.to_excel(Path.joinpath(STAT_PATH,"all_descriptors_без_выбросов.xlsx"))
