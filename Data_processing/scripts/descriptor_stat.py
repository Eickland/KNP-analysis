import pandas as pd
import HumSpectra.utilits as ut

data = pd.read_excel(r"D:\lab\KNP-analysis\Received_data\Statistics\all_descriptors_2025.xlsx")
data.rename(columns={'Проба':'Sample'}, inplace=True)
data['Subclass'] = data['Sample'].apply(lambda x: ut.extract_subclass_from_name(x))

subclass_list = data['Subclass'].unique()

for subclass in subclass_list:
    
    subclass_data = data[data['Subclass'] == subclass]
    subclass_data.describe().to_excel(fr"D:\lab\KNP-analysis\Received_data\Statistics\subclass_statistic\{subclass}_stat.xlsx")