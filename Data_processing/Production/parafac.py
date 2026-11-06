from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from HumSpectra.parafac import EEMDataLoader, ComponentVisualizer
from HumSpectra.sample import SampleCollection, split_by_category
import HumSpectra.utilits as ut
import pandas as pd
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20

from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, PARAFAC_PATH

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(len(collection.samples))
sample_list = []

data = pd.read_excel(r"D:\lab\KNP-analysis\Received_data\Statistics\2024\Documents\Исходные данные за 2024 год.xlsx")

for name, sample in collection.samples.items():
    if '2025' in sample.sample_id and sample.sample_subclass != 'Baikal':
        continue
    
    if sample.sample_id not in data['Sample'].to_list() and sample.sample_subclass != 'Baikal':
        continue
    
    if sample.sample_id == 'ADOM-SLB-B1-4-2025':
        continue
    
    if sample.sample_subclass == 'Mix':
        continue
    
    if sample.sample_subclass == 'Storage 8':
        continue    
    sample_list.append(sample)
    
collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection = collection.from_samples(sample_list)
data = collection.create_flexible_descriptors_table(0.25,sample_attributes=['org_carbon','pH','Eh'])
data.to_csv("2024.csv")
data.to_excel("2024.xlsx")
print(len(collection.samples))
loader = EEMDataLoader()
eem_tensor,em_waves,ex_waves,sp_names,_ = loader.load_eem_from_samples(collection,filter_tags=None)
data_info = loader.get_data_info()

analyzer = ComponentVisualizer(n_components=3)
factors = analyzer.fit_parafac(
    eem_tensor,
    excitation_wavelengths=ex_waves,
    emission_wavelengths=em_waves,
    sample_names=sp_names)

analyzer.plot_component_loadings(normalization='sum',group_type='Subclass')
plt.ylabel('Сумма значений компонентов')
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","component_loadings.png"))

analyzer.plot_component_profiles()
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","component_profiles.png"))

analyzer.plot_fraction_profiles_grouped()
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","fraction_profiles_grouped.png"))

analyzer.plot_all_components_eem(figsize=(5,14))
plt.tight_layout()
plt.savefig(Path.joinpath(PARAFAC_PATH,"graphs","plot_all_components_eem.png"))

analyzer.export_loadings_to_excel(Path.joinpath(PARAFAC_PATH,"data","parafac.xlsx"),normalization='max')