from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from HumSpectra.parafac import EEMDataLoader, ComponentVisualizer
from HumSpectra.sample import SampleCollection, split_by_category
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, PARAFAC_PATH

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0.5)
data.dropna(inplace=True)

MainData = ut.filter_by_iqr_per_group(data,iqr_param=1.5,target_column='Subclass', filter_columns=["Asm_280",'Asm_350','HIX','FIX'])
white_names_list = list(MainData['Sample'].values)

filter_collection = []

for name, sample in collection.samples.items():
    
    if sample.sample_class == 'ADOM':
        
        filter_collection.append(sample)

loader = EEMDataLoader()
eem_tensor,em_waves,ex_waves,sp_names,_ = loader.load_eem_from_samples(collection)
data_info = loader.get_data_info()

analyzer = ComponentVisualizer(n_components=3)
factors = analyzer.fit_parafac(
    eem_tensor,
    excitation_wavelengths=ex_waves,
    emission_wavelengths=em_waves,
    sample_names=sp_names)

analyzer.plot_component_loadings(normalization='percentage')
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

analyzer.export_loadings_to_csv(Path.joinpath(PARAFAC_PATH,"data","parafac.csv"),normalization='max')