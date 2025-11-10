from pathlib import Path
import matplotlib.pyplot as plt

from HumSpectra.parafac import EEMDataLoader, OpticalDataAnalyzer, ComponentVisualizer

wdir = Path.cwd()
fluo_paths = Path.joinpath(wdir,"Initial_data","2025", "Fluorescence")

loader = EEMDataLoader(fluo_paths)
eem_tensor,em_waves,ex_waves,sp_names = loader.load_eem_data(index_col=None)
data_info = loader.get_data_info()

cleaned, mask, residuals, info = loader.robust_detect_outliers(
        eem_tensor, rank=4, threshold_std=3.0, verbose=True)

print(f"Процент выбросов: {info['n_outliers'] / eem_tensor.size * 100:.2f}%")
print(f"Обнаружено выбросов: {info['n_outliers']}")

analyzer = ComponentVisualizer(n_components=4)
factors = analyzer.fit_parafac(
    cleaned,
    excitation_wavelengths=ex_waves,
    emission_wavelengths=em_waves,
    sample_names=sp_names)

save_path = Path.joinpath(Path.cwd(),"Server","Received_data","Parafac")

analyzer.plot_component_loadings(normalization='percentage')
plt.savefig(Path.joinpath(save_path,"graphs","component_loadings.png"))

analyzer.plot_component_profiles()
plt.savefig(Path.joinpath(save_path,"graphs","component_profiles.png"))

analyzer.plot_fraction_profiles_grouped()
plt.savefig(Path.joinpath(save_path,"graphs","fraction_profiles_grouped.png"))

analyzer.export_loadings_to_csv(Path.joinpath(save_path,"data","parafac_2025.csv"),normalization='max')