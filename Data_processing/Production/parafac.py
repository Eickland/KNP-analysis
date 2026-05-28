from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from HumSpectra.parafac import EEMDataLoader, ComponentVisualizer
from HumSpectra.sample import SampleCollection, split_by_category
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20

from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, PARAFAC_PATH

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))

sample_list = []

for name, sample in collection.samples.items():
    
    if sample.sample_subclass == 'Mix':
        continue
    
    if sample.sample_subclass == 'Storage 8':
        continue    
    sample_list.append(sample)
    
collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection = collection.from_samples(sample_list)

print(len(collection.samples))
loader = EEMDataLoader()
eem_tensor,em_waves,ex_waves,sp_names,_ = loader.load_eem_from_samples(collection,filter_tags=['2025'])
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