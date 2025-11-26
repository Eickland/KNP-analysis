import pandas as pd
from pathlib import Path
import sys
from itertools import combinations

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_analysis_with_subclasses, random_forest_analysis_with_subclasses, kmeans_analysis_with_subclasses

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

def get_top_level(first_val, second_val):
    if first_val == 'ADOM':
        return 'ADOM'
    elif second_val == 'Lg':
        return 'Lg'
    else:
        return 'HS'

fluo_path = Path.joinpath(INIT_DATA_2024, "Fluorescence")
uv_path = Path.joinpath(INIT_DATA_2024, "UV","Individual_spectra")
wdir_savepath = Path.joinpath(Path.cwd(),"Server","Received_data","Model")

carbon_data = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'C_org.xlsx'))
carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+'-2024')
geodata = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'GeoData.xlsx'))
geodata.Sample = geodata.Sample.apply(lambda x: x+'-2024')

fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")

parafac_data = pd.read_csv(Path.joinpath(Path.cwd(),"Server","Received_data","Parafac","data","parafac.csv"))
parafac_data.rename(columns={parafac_data.columns[0]:"Sample"},inplace=True)

fl_data = ut.spectra_to_df(fluo_spectra, {
    "Asm_350": fl.asm_350, "Asm_280": fl.asm_280,
    "FIX": fl.fluo_index, "HIX": fl.humin_index
},class_filter=None)

uv_data = ut.spectra_to_df(uv_spectra, {
    "E2E3": uv.e2_e3, "epsilon": uv.single_density
},class_filter=None)

MainData_init = (uv_data.merge(carbon_data, on="Sample")
                .assign(Suva=lambda x: 100*x.epsilon / x.C_org)
                .drop(columns=["epsilon"])
#                .merge(geodata[["Sample", "pH", "eH"]], on="Sample")
                .merge(fl_data, on=["Sample", "Class", "Subclass"])
                .merge(parafac_data,on=["Sample", "Subclass"]))

numeric_cols = MainData_init.select_dtypes(include=['float']).columns
MainData_init[numeric_cols] = MainData_init[numeric_cols].map(lambda x: float(f"{x:.4g}"))
MainData_init.Sample = MainData_init.Sample.apply(lambda x: x+"-2025")

MainData = ut.filter_by_iqr_per_group(MainData_init,iqr_param=1.5,target_column='Subclass', filter_columns=["Suva","Asm_280","C_org"])
MainData.set_index(["Class","Subclass","Sample"],inplace=True)

feature_columns = MainData.columns.to_list()
#feature_columns = ['Component_1','Component_2','Component_3','Component_4','E2E3','Asm_280','Asm_350','HIX','FIX']

feature_columns_kmeans = feature_columns
feature_columns_lda = feature_columns
feature_columns_rf = feature_columns

acc_lda = []
acc_rf = []
num_fearure_max = 2
combinations_list = list(combinations(feature_columns, num_fearure_max))

new_level_values = [
    get_top_level(first_val, second_val) 
    for first_val, second_val, _ in MainData.index
]

# Добавляем новый уровень
new_index = pd.MultiIndex.from_arrays(
    [new_level_values] + [MainData.index.get_level_values(i) for i in range(MainData.index.nlevels)],
    names=['Overclass', 'Class', 'Subclass','Sample']
)

MainData = MainData.set_index(new_index)


res_lda = lda_analysis_with_subclasses(MainData[['E2E3', 'C_org', 'Asm_350', 'Asm_280']],save=True,vif_threshold = 100000,index_level=0,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{'E2E3', 'C_org', 'Asm_350', 'Asm_280'}_2024.html")
                                        )
                                    )

for comb in combinations_list:
    
    comb = list(comb)
    res_lda = lda_analysis_with_subclasses(MainData[comb],save=False,vif_threshold = 100000,index_level=0,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{comb}_2024.html")
                                        )
                                    )
    """
    res_kmeans = kmeans_analysis_with_subclasses(MainData[comb],save=True,n_clusters=2,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Clustering","KM",f"KM_{comb}_2024.html")
                                        )
                                    )
    

    res_rf = random_forest_analysis_with_subclasses(MainData[feature_columns_rf],save=False,target_column='Class',
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Classification","RF",f"RF_{comb}.html")
                                        )
                                    )
    """
    acc_lda.append(res_lda[-4])
#    acc_rf.append(res_rf[-1])
    
data = pd.DataFrame()

data['LDA'] = acc_lda
#data['RF'] = acc_rf
data['Feature'] = combinations_list

data.sort_values('LDA',ascending=False).to_excel(Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model",f"Output_{num_fearure_max}.xlsx"))