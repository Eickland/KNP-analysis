import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_classifaction, random_forest_classification
import matplotlib as mpl
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

# Установка размеров шрифтов для всех элементов

def create_hierarchical_scatter(df, class_col, subclass_col, x_col, y_col,s=70):
    """
    Использует HSV цветовое пространство для лучшего контроля насыщенности
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    classes = df[class_col].unique()
    subclasses = df[subclass_col].unique()
    
    # Насыщенные цвета в HSV пространстве
    base_hues = np.linspace(0, 0.9, len(classes))  # Равномерное распределение оттенков
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'X']
    
    for i, class_val in enumerate(classes):
        class_data = df[df[class_col] == class_val]
        hue = base_hues[i]
        
        # Создаем градиент варьируя насыщенность (S) и значение (V) в HSV
        saturations = np.linspace(0.8, 0.4, len(subclasses))  # От насыщенного к менее насыщенному
        values = np.linspace(0.9, 0.7, len(subclasses))       # От яркого к темному
        
        for j, subclass_val in enumerate(subclasses):
            subclass_data = class_data[class_data[subclass_col] == subclass_val]
            
            if len(subclass_data) > 0:
                # Конвертируем HSV в RGB
                hsv_color = (hue, saturations[j], values[j])
                rgb_color = mpl.colors.hsv_to_rgb(hsv_color) # type: ignore
                
                ax.scatter(subclass_data[x_col], subclass_data[y_col],
                          color=rgb_color,
                          marker=markers[j % len(markers)],
                          s=s,
                          label=f'{class_val} - {subclass_val}',
                          edgecolor='black', 
                          linewidth=1.0,
                          alpha=0.85)
    
    #ax.legend(bbox_to_anchor=(1, 1), loc='upper right')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax

def get_top_level(first_val, second_val):
    if first_val == 'ADOM':
        return 'ADOM'
    elif second_val == 'Lg':
        return 'Lg'
    elif first_val == 'Baikal':
        return 'Baikal'
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
                .merge(parafac_data,on=["Sample", "Subclass",'Class'])
)

numeric_cols = MainData_init.select_dtypes(include=['float']).columns
MainData_init[numeric_cols] = MainData_init[numeric_cols].map(lambda x: float(f"{x:.4g}"))
#MainData_init.Sample = MainData_init.Sample.apply(lambda x: x+"-2024")

MainData = ut.filter_by_iqr_per_group(MainData_init,iqr_param=1.5,target_column='Subclass', filter_columns=["Suva","Asm_280","C_org"])
print(MainData)
#MainData = MainData[MainData.Class != 'Lst']
MainData.set_index(["Class","Subclass","Sample"],inplace=True)

feature_columns = MainData.columns.to_list()
#feature_columns = ['Component_1','Component_2','Component_3','Component_4','E2E3','Asm_280','Asm_350','HIX','FIX']

feature_columns_kmeans = feature_columns
feature_columns_lda = feature_columns
feature_columns_rf = feature_columns

acc_lda = []
acc_rf = []
num_fearure_max = 1
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

feature_comb = ['Asm_280', 'FIX', 'Component_2']

res_lda = lda_classifaction(MainData[feature_comb],index_level=1,
                                        output_html_path = str(
                                            Path.joinpath(
                                                Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{feature_comb}.html")
                                            )
                                        )

res_rf = random_forest_classification(MainData[feature_comb],index_level=1,
                                        output_html_path = str(
                                            Path.joinpath(
                                                Path.cwd(),"Server","Received_data","Model","Classification","RF",f"RF_{feature_comb}.html")
                                            )
                                        )

"""
feature_custom_listD2 = [['E2E3', 'Suva', 'Asm_280', 'FIX'],['Asm_350', 'Asm_280', 'Component_1', 'Component_2'],['Suva', 'Asm_280', 'FIX', 'Component_2'],['C_org', 'Asm_350', 'Asm_280', 'Component_2'],
                         ['Suva', 'Asm_350', 'Asm_280', 'Component_2'],['Asm_350', 'Asm_280', 'Component_2'],['Asm_280', 'FIX', 'Component_2'],['C_org', 'Asm_350', 'Component_1'],
                         ['C_org', 'Suva', 'FIX'],['Asm_280', 'HIX'],['Asm_350', 'Asm_280'],['Asm_280', 'FIX'],['Component_2', 'Asm_280'],['C_org', 'Suva', 'Asm_280'],['C_org', 'Suva', 'Asm_280', 'FIX']]

for feature_comb in [['Asm_280', 'FIX', 'Component_2']]:
    res_lda = lda_classifaction(MainData[feature_comb],index_level=1,threshold = 10000,
                                        output_html_path = str(
                                            Path.joinpath(
                                                Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{feature_comb}_oldclass_2024.html")
                                            )
                                        )

for comb in combinations_list:
    
    comb = list(comb)
    res_lda = lda_classifaction(MainData[comb],vif_threshold = 1,index_level=1,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Classification","LDA",f"LDA_{comb}_sub_2024.html")
                                        )
                                    )
    res_kmeans = kmeans_analysis_with_subclasses(MainData[comb],save=True,n_clusters=2,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Clustering","KM",f"KM_{comb}_2024.html")
                                        )
                                    )
    
    res_rf = random_forest_classification(MainData[feature_columns_rf],index_level=0,
                                    output_html_path = str(
                                        Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model","Classification","RF",f"RF_{comb}.html")
                                        )
                                    )
    
    acc_lda.append(res_lda[-4])
    acc_rf.append(res_rf[-1])
    
data = pd.DataFrame()

data['LDA_macro'] = acc_lda
data['RF'] = acc_rf
data['Feature'] = combinations_list

data.sort_values('LDA_macro',ascending=False).to_excel(Path.joinpath(
                                            Path.cwd(),"Server","Received_data","Model",f"Output_{num_fearure_max}.xlsx"))

"""