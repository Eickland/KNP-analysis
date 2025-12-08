from pathlib import Path
import matplotlib.pyplot as plt
import sys
from typing import List, Any
import pandas as pd
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
import random
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

FluoPlotEnable = True
UVPlotEnable = False

plt.rcParams['figure.dpi'] = 300

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

date_list = ["2024"]
date_folder_list = [INIT_DATA_2024]

carbon_data = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'C_org.xlsx'))
carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+'-2024')

for i, folder in enumerate(date_folder_list):
    
    fluo_path = Path.joinpath(folder, "Fluorescence")
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")
    
    fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
    uv_abs_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
    
    uv_toc_spectra = uv.add_toc_to_spectra(uv_abs_spectra,carbon_data)
    
    uv_class_spectra = ut.split_dataframes_by_class(uv_toc_spectra,type='class')
    fluo_subclass_spectra = ut.split_dataframes_by_class(fluo_spectra,type= 'class')
    
    if i == 1:
        uv_refl_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")
    fig,ax = plt.subplots(1,1,figsize=(7,7),dpi=300)
    for fluo_subclass in fluo_subclass_spectra: 
        random_element = random.sample(fluo_subclass, 1)[0]
        if random_element.attrs['subclass'] in ['Lg','Lst','SHA','SFA','SHF','PFA','PHF','PHA','CHF','CHA']:
        
            fl_spectra = fl.cut_spectra(random_element,255,450,255,600)
            fl_spectra = ut.normilize_by_max(fl_spectra)
            fl_spectra_280 = fl.cut_raman_spline(em_wvs=fl_spectra.index.to_numpy(),fl=fl_spectra[280].to_numpy(),exc_wv=280)
            ax.plot(fl_spectra.index.to_numpy(),fl_spectra_280,label=random_element.attrs['name'])
            
            ax.set_title('Спектр флуоресценции при 280 нм длины волны возбуждения')
            ax.set_ylabel('Интенсивность')
            ax.set_xlabel('Длина волны испускания, нм')
            ax.set_ylim((0.0,1.1))
        else:
            continue
    plt.legend()
    plt.tight_layout()
    plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","Fluorescence","280_nm",f"Карты+байкал")
                        )
    plt.close()
    
    if FluoPlotEnable:
    
        for fl_spectra in fluo_spectra:
            
            fl_spectra = fl.cut_spectra(fl_spectra,255,450,255,600)
            fl.plot_heat_map(fl_spectra.T)
            
            plt.tight_layout()
            plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","Fluorescence","Individual_spectra",f"{fl_spectra.attrs['name']}")
                        )
            plt.close()
        
    fig,ax = plt.subplots(1,1,figsize=(7,7),dpi=300) 
    for uv_class_list in uv_class_spectra:
        
        random_elements = random.sample(uv_class_list, 1)
        if random_elements[0].attrs['class'] in ['ADOM','Baikal']:
            
            continue
        else:
            for uv_spectra in random_elements:
                
                uv_spectra = uv_spectra.iloc[20:]
                
                uv.plot_uv(uv_spectra,norm_by_TOC=True,name=uv_spectra.attrs['name'],ax=ax,title=False)
                ax.set_title('Нормированный спектр поглощения')
                ax.set_xlabel('Длина волны, нм')
    plt.legend()    
    plt.tight_layout()  
    plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Grouped_spectra",f"Нормированный спектр поглощения_2") # type: ignore
                        )
    plt.close()
    
    if UVPlotEnable:
             
        for uv_spectra in uv_abs_spectra:
            
            uv_spectra = uv_spectra.iloc[30:]
            
            uv.plot_uv(uv_spectra)
            
            plt.tight_layout()
            plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Individual_spectra",f"{uv_spectra.attrs['name']}")
                        )
            plt.close()
                
        
        if i == 1:
            
            for uv_spectra in uv_refl_spectra: # type: ignore
                
                uv.plot_uv(uv_spectra)
                
                plt.tight_layout()
                plt.savefig(
                    Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Reflection","Individual_spectra",f"{uv_spectra.attrs['name']}")
                            )
                plt.close()
        