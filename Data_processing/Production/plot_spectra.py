from pathlib import Path
import matplotlib.pyplot as plt
import sys

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

date_list = ["2024","2025"]
date_folder_list = [INIT_DATA_2024,INIT_DATA_2025]

for i, folder in enumerate(date_folder_list):
    
    fluo_path = Path.joinpath(folder, "Fluorescence")
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")
    
    fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
    uv_abs_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
    
    if i == 1:
        uv_refl_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")
    
    for fl_spectra in fluo_spectra:
        
        fl_spectra = fl.cut_spectra(fl_spectra,255,450,255,600)
        fl.plot_heat_map(fl_spectra)
        
        plt.tight_layout()
        plt.savefig(
            Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","Fluorescence","Individual_spectra",f"{fl_spectra.attrs['name']}")
                    )
        plt.close()
        
    for uv_spectra in uv_abs_spectra:
        
        uv.plot_uv(uv_spectra)
        
        plt.tight_layout()
        plt.savefig(
            Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Individual_spectra",f"{uv_spectra.attrs['name']}")
                    )
        plt.close()
    
    if i == 1:
         
        for uv_spectra in uv_refl_spectra:
            
            uv.plot_uv(uv_spectra)
            
            plt.tight_layout()
            plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Reflection","Individual_spectra",f"{uv_spectra.attrs['name']}")
                        )
            plt.close()
        