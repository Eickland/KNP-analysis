import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

str_years = [str(year) for year in range(2000, 2025 + 1)]
articles = [159,163,198,436,295,318,328,442,482,543,712,800,903,993,1170,1300,1450,1480,1650,1870,2060,2550,2920,3520,3830,5600]

plt.rcParams['axes.titlesize'] = 24
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 20

sns.barplot(x=str_years,y=articles)
plt.title('К-во опубликованных статей по ключевым словам\n"dissolved organic matter" AND "machine learning"')
plt.ylabel('К-во публикаций')
plt.xlabel('Гойда')
plt.tight_layout()
plt.show()