from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
from HumSpectra.parafac import EEMDataLoader, ComponentVisualizer

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, PARAFAC_PATH

# Установка размеров шрифтов для всех элементов
plt.rcParams.update({
    'font.size': 18,                    # Базовый размер шрифта
    'axes.titlesize': 20,               # Заголовок осей
    'axes.labelsize': 18,               # Метки осей
    'xtick.labelsize': 16,              # Метки на оси X
    'ytick.labelsize': 16,              # Метки на оси Y
    'legend.fontsize': 16,              # Легенда
    'figure.titlesize': 20              # Заголовок фигуры
})



fluo_paths_2024 = Path.joinpath(INIT_DATA_2024, "Fluorescence")
fluo_paths_2025 = Path.joinpath(INIT_DATA_2025, "Fluorescence")

loader = EEMDataLoader()
eem_tensor,em_waves,ex_waves,sp_names = loader.load_eem_data(data_folders=[fluo_paths_2024,fluo_paths_2025])
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