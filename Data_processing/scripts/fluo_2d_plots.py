import HumSpectra.fluorescence as fl
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
from Data_processing.Utilits.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

search_list = []

fluo_path = Path.joinpath(INIT_DATA_2025, "Fluorescence")
print(fluo_path)
fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
print(len(fluo_spectra))
for fluo in fluo_spectra:
    print(fluo.attrs['name'])
    if fluo.attrs['name'] in 'ADOM-SLB-B-2-2025' or fluo.attrs['name'] in 'ADOM-SL3-1-2025':
        search_list.append(fluo)
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Initial_data\Fluorescence\ADOM-SLB-B-2-2025.csv"
                                   ,remove_raman=True))
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Initial_data\Fluorescence\ADOM-SL3-1-2025.csv"
                                   ,remove_raman=True))
#search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data\B2_K3_20.csv",remove_raman=True))     
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data\B2_K3_40.csv"
                                   ,remove_raman=True))     
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data\B2_K3_60.csv",remove_raman=True))     
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data\B2_K3_100.csv",remove_raman=True))     
search_list.append(fl.read_fluo_3d(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\raw_data\B2_K3_150.csv",remove_raman=True))     

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
for spectra in search_list:
    
    fl.plot_2d(spectra/spectra.loc[400],350,ax=ax,norm=False)
    plt.legend()
plt.show()