import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from pandas import DataFrame
from typing import List
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from io import StringIO
import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             mean_squared_error, r2_score, mean_absolute_error)
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut

wdir_date = Path.cwd().parent / "Initial_data" / "2025"
fluo_path = Path.joinpath(wdir_date, "Fluorescence")
uv_path = Path.joinpath(wdir_date, "UV","Individual spectra")
wdir_savepath = Path.joinpath(Path.cwd().parent,"Server","Received_data","Clustering")

carbon_data = pd.read_excel(Path.joinpath(wdir_date, 'C_org.xlsx'))
geodata = pd.read_excel(Path.joinpath(wdir_date, 'GeoData.xlsx'))

fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")

parafac_data = pd.read_csv(Path.joinpath(Path.cwd().parent,"Server","Received_data","Parafac","parafac_2025.csv"))
parafac_data.rename(columns={parafac_data.columns[0]:"Sample"},inplace=True)

fl_data = ut.spectra_to_df(fluo_spectra, {
    "Asm_350": fl.asm_350, "Asm_280": fl.asm_280,
    "FIX": fl.fluo_index, "HIX": fl.humin_index
})

uv_data = ut.spectra_to_df(uv_spectra, {
    "E2E3": uv.e2_e3, "epsilon": uv.epsilon
})

MainData_init = (uv_data.merge(carbon_data, on="Sample")
                .assign(Suva=lambda x: 100*x.epsilon / x.C_org)
                .drop(columns=["epsilon"])
                .merge(geodata[["Sample", "pH", "eH"]], on="Sample")
                .merge(fl_data, on=["Sample", "Class", "Subclass"])
                .merge(parafac_data,on="Sample"))

numeric_cols = MainData_init.select_dtypes(include=['float']).columns
MainData_init[numeric_cols] = MainData_init[numeric_cols].map(lambda x: float(f"{x:.4g}"))
MainData_init.Sample = MainData_init.Sample.apply(lambda x: x+"-2025")

MainData = ut.delete_eject_iqr(MainData_init,iqr_param=2.3,multi_index=False,columns=["Suva","Asm_280","C_org"])

feature_columns = MainData.columns.to_list()

MainData.set_index(["Class","Subclass","Sample"],inplace=True)