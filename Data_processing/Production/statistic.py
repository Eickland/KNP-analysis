import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

date_list = ["2024","2025"]
date_folder_list = [INIT_DATA_2024,INIT_DATA_2025]

plt.rcParams['figure.dpi'] = 300

for i, folder in enumerate(date_folder_list):
    
    fluo_path = Path.joinpath(folder, "Fluorescence")
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")

    carbon_data = pd.read_excel(Path.joinpath(folder, 'C_org.xlsx'))
    carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+'-'+date_list[i])

    geodata = pd.read_excel(Path.joinpath(folder, 'GeoData.xlsx'))
    geodata.Sample = geodata.Sample.apply(lambda x: x+'-'+date_list[i])

    fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
    uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")

    fl_data = ut.spectra_to_df(fluo_spectra, {
        "Asm_350": fl.asm_350, "Asm_280": fl.asm_280,
        "FIX": fl.fluo_index, "HIX": fl.humin_index
    })

    uv_data = ut.spectra_to_df(uv_spectra, {
        "E2E3": uv.e2_e3, "epsilon": uv.single_density
    })
    
    parafac_data = pd.read_csv(Path.joinpath(PARAFAC_PATH,"data","parafac.csv"))

    MainData = (uv_data.merge(carbon_data, on="Sample")
                    .assign(Suva=lambda x: 100*x.epsilon / x.C_org)
                    .drop(columns=["epsilon"])
                    .merge(geodata[["Sample", "pH", "eH"]], on="Sample")
                    .merge(fl_data, on=["Sample", "Class", "Subclass"])
                    .merge(parafac_data,on=["Sample", "Subclass"])
                    )

    numeric_cols = MainData.select_dtypes(include=['float']).columns
    MainData[numeric_cols] = MainData[numeric_cols].map(lambda x: float(f"{x:.4g}"))
    
    MainData.set_index(["Class","Subclass","Sample"],inplace=True)
    
    MainData.to_excel(Path.joinpath(STAT_PATH,f'{date_list[i]}',
                                    'Documents',f'Исходные данные за {date_list[i]} год.xlsx'))
    
    MainData = ut.filter_by_iqr_per_group(MainData.reset_index(),'Subclass',iqr_param=3,
                                          filter_columns=['pH','eH','C_org'])
    
    MainData.to_excel(Path.joinpath(STAT_PATH,f'{date_list[i]}',
                                    'Documents',f'Исходные данные без выбросов за {date_list[i]} год.xlsx'))
    
    sns.pairplot(MainData, hue = 'Subclass')
    plt.savefig(
        Path.joinpath(STAT_PATH,f'{date_list[i]}',
                      'Graphs',f'pairplot_{date_list[i]}.png')
                )
    plt.close()
    

    
    if date_list[i] == '2025':
        
        for uvs in uv_spectra:
            print(uvs.attrs['name'])
                    
        print(uv_data.head(50))
        
        print('carbon_data'+ f'{carbon_data['Sample'].isin(['ADOM-SL3-2-2025']).any()}')
        print('geodata'+ f'{geodata['Sample'].isin(['ADOM-SL3-2-2025']).any()}')
        print('fl_data'+ f'{fl_data['Sample'].isin(['ADOM-SL3-2-2025']).any()}')
        print('parafac_data'+ f'{parafac_data['Sample'].isin(['ADOM-SL3-2-2025']).any()}')
        print('uv_data'+ f'{uv_data['Sample'].isin(['ADOM-SL3-2-2025']).any()}')