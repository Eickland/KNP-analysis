from HumSpectra.sample import Sample, SampleCollection
import HumSpectra.fluorescence as fl
import HumSpectra.utilits as ut
import HumSpectra.ultraviolet as uv
from HumSpectra.hum_statistic import lda_classification, random_forest_classification

from pathlib import Path
import numpy as np
import sys
import pandas as pd
from itertools import combinations

current_dir = Path(__file__).parent.parent
neighbor_dir = current_dir / 'Production'
sys.path.append(str(neighbor_dir))

from statistic import scatter_plot_default

water_research_data_dir = r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\result_data\result.csv"
adom_2025_data_dir = r"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Received_data\Statistics\descriptors.xlsx"

adom_2025_preprocess_data = pd.read_excel(adom_2025_data_dir)
print(f'Всего образцов: {len(adom_2025_preprocess_data)}')
print(adom_2025_preprocess_data)
adom_2025_data = ut.filter_by_iqr_per_group(adom_2025_preprocess_data,iqr_param=1.5,
                                            target_column='Subclass', filter_columns=['asm_350','asm_280','FIX','HIX'])
print(f'Всего образцов после удаления выбросов: {len(adom_2025_data)}')

water_research_data = pd.read_csv(water_research_data_dir)
water_research_data['Class'] = 'Mix'
water_research_data['Subclass'] = water_research_data['Class']
default_columns = water_research_data.columns

data = pd.concat([water_research_data,adom_2025_data[default_columns]])

scatter_plot_default(data,path=r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\scatterplots\Subclass",
                     target_column='Subclass',save=True,label_class='Mix')

data.set_index(["Class","Subclass","Sample"],inplace=True)

feature_comb = ['Asm_280', 'Asm_350', 'HIX','FIX']

num_fearure_list = [1,2,3,4]
num_iterations = 0
k = 1

accuracy_lda = []
weight_accuracy_lda=[]
accuracy_rf = []
weight_accuracy_rf=[]
combinations_2Dlist = []
count_feature_list = []        

for num_fearure_max in num_fearure_list:
    num_iterations += len(list(combinations(feature_comb, num_fearure_max)))

for num_fearure_max in num_fearure_list:

    combinations_list = list(combinations(feature_comb, num_fearure_max))

    for iteration in combinations_list:
        
        print(f'{k}/{num_iterations}')

        res_lda = lda_classification(data[list(iteration)],index_level=0,
                            output_html_path=fr"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\models\lda\{iteration}.html",
                            external_validation=False)
        weight_accuracy_lda.append(res_lda[-4])
        accuracy_lda.append(res_lda[-5])        
        
        
        res_rf = random_forest_classification(data[list(iteration)],index_level=0,
                            output_html_path=fr"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\models\rf\{iteration}.html",
                            external_validation=False)
        
        weight_accuracy_rf.append(res_rf[-4])
        accuracy_rf.append(res_rf[-5])
        
        combinations_2Dlist.append(iteration)
        count_feature_list.append(num_fearure_max)
        k+=1
        
accuracy_data = pd.DataFrame()

accuracy_data['LDA_weight_accuracy'] = weight_accuracy_lda
accuracy_data['LDA_accuracy'] = accuracy_lda    
accuracy_data['RF_weight_accuracy'] = weight_accuracy_rf
accuracy_data['RF_accuracy'] = accuracy_rf
accuracy_data['feature'] = combinations_2Dlist
accuracy_data['num_feature'] = count_feature_list

accuracy_data.sort_values('LDA_weight_accuracy',ascending=False).to_excel(r"C:\Users\Kirill\Desktop\Lab\KNP-work-directory\article_water_research\models\models_data.xlsx")                                