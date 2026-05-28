import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams["legend.fontsize"] = 17

optical_df=pd.read_csv(r"D:\lab\KNP-analysis\Received_data\Statistics\all_descriptors.csv")
optical_df = optical_df[optical_df['sample_id'].str.contains('2025')]
optical_df = optical_df[optical_df['Subclass'] != 'Storage 8']
optical_df.drop(columns=['lambda','Sample','pH','org_carbon','Eh'],inplace=True)
optical_df.dropna(inplace=True)
optical_df.rename(columns={'sample_id':'sample_name','Class':'class'},inplace=True)
print(optical_df.columns)
sns.pairplot(data=optical_df[['HIX','FIX', 'Asm_280','Asm_350', 'Subclass']], 
             hue='Subclass')
plt.savefig(r"D:\lab\KNP-analysis\Received_data\Statistics\Correlations\Graphs\fluo_pair_plot.png")
plt.close()