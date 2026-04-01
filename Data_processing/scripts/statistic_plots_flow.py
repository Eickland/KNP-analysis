from pathlib import Path
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import HumSpectra.utilits as ut

import os

plt.rcParams['axes.titlesize'] = 24
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['axes.labelsize'] = 20

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
current_dir = Path(__file__).parent.parent
neighbor_dir = current_dir / 'Production'
sys.path.append(str(neighbor_dir))
from statistic import hist_plot_default,swarm_plot_default, feature_names_translate_dict
from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, CLASSIFICATION_PATH

yandex_disk_path = os.getenv('YANDEX_DISK_PATH')
if yandex_disk_path is None:
    raise KeyError('Ошибка подключения яндекс диска')

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0.25)
data.dropna(inplace=True)

MainData = ut.filter_by_iqr_per_group(data,iqr_param=1.5,target_column='Subclass', filter_columns=["org_carbon",'Suva'])
MainData = MainData[MainData['Subclass'] != 'Storage 1']

hist_plot_default(MainData,path=str(Path(yandex_disk_path) / "article_water_research" / "plots" /"histplots"),
                     target_column='Subclass',save=True)

swarm_plot_default(MainData,path=str(Path(yandex_disk_path) / "article_water_research" / "plots" /"swarmplots"),
                     target_column='Subclass',save=True)



