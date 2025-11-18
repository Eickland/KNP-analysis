import pandas as pd
from pathlib import Path
import sys

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_analysis_with_subclasses, random_forest_analysis_with_subclasses, kmeans_analysis_with_subclasses

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

fluo_path = Path.joinpath(INIT_DATA_2025, "Fluorescence")
uv_path = Path.joinpath(INIT_DATA_2025, "UV","Individual_spectra")
wdir_savepath = Path.joinpath(Path.cwd(),"Server","Received_data","Model")

carbon_data = pd.read_excel(Path.joinpath(INIT_DATA_2025, 'C_org.xlsx'))
geodata = pd.read_excel(Path.joinpath(INIT_DATA_2025, 'GeoData.xlsx'))

fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")

parafac_data = pd.read_csv(Path.joinpath(Path.cwd(),    "Server","Received_data","Parafac","data","parafac.csv"))
parafac_data.rename(columns={parafac_data.columns[0]:"Sample"},inplace=True)

fl_data = ut.spectra_to_df(fluo_spectra, {
    "Asm_350": fl.asm_350, "Asm_280": fl.asm_280,
    "FIX": fl.fluo_index, "HIX": fl.humin_index
})

uv_data = ut.spectra_to_df(uv_spectra, {
    "E2E3": uv.e2_e3, "epsilon": uv.single_density
})

MainData_init = (uv_data.merge(carbon_data, on="Sample")
                .assign(Suva=lambda x: 100*x.epsilon / x.C_org)
                .drop(columns=["epsilon"])
                .merge(geodata[["Sample", "pH", "eH"]], on="Sample")
                .merge(fl_data, on=["Sample", "Class", "Subclass"])
                .merge(parafac_data,on=["Sample", "Subclass"]))

numeric_cols = MainData_init.select_dtypes(include=['float']).columns
MainData_init[numeric_cols] = MainData_init[numeric_cols].map(lambda x: float(f"{x:.4g}"))
MainData_init.Sample = MainData_init.Sample.apply(lambda x: x+"-2025")

MainData = ut.delete_eject_iqr(MainData_init,iqr_param=3,multi_index=False,columns=["Suva","Asm_280","C_org"])
MainData.set_index(["Class","Subclass","Sample"],inplace=True)

#feature_columns = MainData.columns.to_list()[:-8]
feature_columns = ['Component_1','Component_2','Component_3','Component_4','E2E3','Asm_280','Asm_350','HIX','FIX']

feature_columns_kmeans = feature_columns
feature_columns_lda = feature_columns
feature_columns_rf = feature_columns


res_lda = lda_analysis_with_subclasses(MainData[feature_columns_lda],save=True,vif_threshold = 100000,
                                   output_html_path = str(
                                       Path.joinpath(
                                           Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{feature_columns_lda}.html")
                                       )
                                   )

res_kmeans = kmeans_analysis_with_subclasses(MainData[feature_columns_kmeans],save=True,
                                   output_html_path = str(
                                       Path.joinpath(
                                           Path.cwd(),"Server","Received_data","Model","Clustering","KM",f"KM_{feature_columns_kmeans}.html")
                                       )
                                   )

res_rf = random_forest_analysis_with_subclasses(MainData[feature_columns_rf],save=True,
                                   output_html_path = str(
                                       Path.joinpath(
                                           Path.cwd(),"Server","Received_data","Model","Classification","RF",f"RF_{feature_columns_rf}.html")
                                       )
                                   )