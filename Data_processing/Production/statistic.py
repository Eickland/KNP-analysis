import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from itertools import combinations

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

date_list = ["2024","2025"]
date_folder_list = [INIT_DATA_2024,INIT_DATA_2025]

PairPlotEnable = False
BoxPlotEnable = False
KdePlotEnable = False
HistPlotEnable = False
ImPlotEnable = False
ScatterPlotEnable = True

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
    }, class_filter=None)

    uv_data = ut.spectra_to_df(uv_spectra, {
        "E2E3": uv.e2_e3, "epsilon": uv.single_density, "lambda": uv.lambda_UV
    }, class_filter=None)
    
    parafac_data = pd.read_csv(Path.joinpath(PARAFAC_PATH,"data","parafac.csv"))
    """
    MainData = (uv_data.merge(carbon_data, on="Sample")
                    .assign(Suva=lambda x: 100*x.epsilon / x.C_org)
                    .merge(geodata[["Sample", "pH", "eH"]], on="Sample")
                    .merge(fl_data, on=["Sample", "Class", "Subclass"])
                    .merge(parafac_data,on=["Sample", "Subclass"])
                    )
    """
    MainData = (uv_data.merge(carbon_data, on="Sample")
                    .assign(Suva=lambda x: 100*x.epsilon / x.C_org).merge(fl_data, on=["Sample", "Class", "Subclass"])
                    .merge(parafac_data,on=["Sample", "Subclass"])
                    )
    
    numeric_cols = MainData.select_dtypes(include=['float']).columns
    MainData[numeric_cols] = MainData[numeric_cols].map(lambda x: float(f"{x:.4g}"))
    
    MainData.set_index(["Class","Subclass","Sample"],inplace=True)
    
    MainData.to_excel(Path.joinpath(STAT_PATH,f'{date_list[i]}',
                                    'Documents',f'Исходные данные за {date_list[i]} год.xlsx'))
    
    MainData = ut.filter_by_iqr_per_group(MainData.reset_index(),'Class',iqr_param=3,
                                          filter_columns=['Asm_350','lambda'])
    
    MainData.to_excel(Path.joinpath(STAT_PATH,f'{date_list[i]}',
                                    'Documents',f'Исходные данные без выбросов за {date_list[i]} год.xlsx'))
    
    if PairPlotEnable:
        
        fig, axes = plt.subplots(1,1,figsize=(12,12), dpi=600)
    
        sns.pairplot(MainData, hue = 'Class')
        plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs',f'pairplot_{date_list[i]}.png')
                    )
        plt.close()
    
    if BoxPlotEnable:
    
        for feature in numeric_cols:
            
            fig, axes = plt.subplots(1,1,figsize=(8,8))
            
            box = sns.boxplot(data = MainData.reset_index(), hue='Class', y=feature, ax=axes)
            ut.add_median_labels(box)
            
            plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs','boxplot',f'boxplot_{feature}_{date_list[i]}.png')
                    )
            plt.close()
            
            
    if KdePlotEnable:
        
        combinations_list = list(combinations(numeric_cols, 2))
    
        for feature_1, feature_2 in combinations_list:
            
            fig, axes = plt.subplots(1,1,figsize=(8,8))
            
            sns.kdeplot(data = MainData.reset_index(), hue='Class', y=feature_2,x = feature_1, ax=axes, alpha = 0.5)
            
            plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs','kdeplot',f'kdeplot_{feature_1} vs {feature_2}_{date_list[i]}.png')
                    )
            plt.close()

    if HistPlotEnable:
    
        for feature in numeric_cols:
            
            fig, axes = plt.subplots(1,1,figsize=(8,8))
            #MainData.reset_index(inplace=True)
            MainData = MainData[MainData['Class'] != 'Baikal']
            sns.histplot(data = MainData, hue='Subclass', x=feature, ax=axes,
                         multiple="stack",palette="rainbow",edgecolor=".3",linewidth=.5,log_scale=False,)
            
            plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs','histplot',f'histplot_{feature}_{date_list[i]}.png')
                    )
            plt.close()
            
    if ImPlotEnable:
        
        combinations_list = list(combinations(numeric_cols, 2))
    
        for feature_1, feature_2 in combinations_list:
                        
            sns.lmplot(data = MainData.reset_index(), hue='Class', y=feature_1,x = feature_2)
            
            plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs','implot',f'implot_{feature_1} vs {feature_2}_{date_list[i]}.png')
                    )
            plt.close()
            
            
    if ScatterPlotEnable:
        
        combinations_list = list(combinations(numeric_cols, 2))
    
        for feature_1, feature_2 in combinations_list:
            
            fig, axes = plt.subplots(1,1,figsize=(8,8))
            sns.scatterplot(data = MainData.reset_index(), hue='Class', y=feature_1,x = feature_2, ax=axes,palette='rainbow',style='Subclass')
            
            plt.savefig(
            Path.joinpath(STAT_PATH,f'{date_list[i]}',
                        'Graphs','scatterplot',f'scatterplot_{feature_1} vs {feature_2}_{date_list[i]}.png')
                    )
            plt.close()