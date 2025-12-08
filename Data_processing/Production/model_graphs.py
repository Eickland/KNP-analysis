import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Настройки стиля
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


model_data = pd.read_excel(Path.joinpath(Path.cwd(),"Server","Received_data","Model",f"Model_accuracy_cross-val.xlsx"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = model_data
# Загрузите ваши данные (предполагая, что они в CSV формате)
# df = pd.read_csv('ваш_файл.csv')
plt.rcParams.update({
    'font.size': 20,                    # Базовый размер шрифта
})

# Дополнительный вариант: раздельные графики с более детальной визуализацией
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 10))

# 1. Гистограмма LDA средней точности
sns.histplot(data=df,x='LDA_weight_accuracy', color='blue',ax=axes2[0],bins=100,alpha=0.6,label='LDA')
axes2[0].set_xlabel('Средневзвешенная точность')
axes2[0].set_ylabel('Частота')
#axes2[0].set_title('Гистограмма средневзвешанной точности')


# 3. Гистограмма RF средней точности
sns.histplot(data=df,x='RF_weight_accuracy', color='orange',ax=axes2[0],bins=100,alpha=0.6,label='RF')
axes2[0].set_xlim(0.5,1)



# Можно агрегировать данные по количеству признаков
lda_grouped = df.groupby('num_feature').agg({
    'ADOM_accuracy_lda': 'mean',
    'ADOM_accuracy_lda_CV': 'mean'
}).reset_index()

# Сортируем по количеству признаков
lda_grouped = lda_grouped.sort_values('num_feature')

# Зависимость точности LDA ADOM от числа признаков
axes2[1].plot(lda_grouped['num_feature'], lda_grouped['ADOM_accuracy_lda'], 
              'r-', linewidth=2, label='ЛДА')


#axes2[1].set_title('LDA: Гистограмма средней точности и зависимость точности ADOM от числа признаков')
axes2[1].legend(loc='upper left')
axes2[1].legend(loc='upper right')

# Агрегируем данные по количеству признаков для RF
rf_grouped = df.groupby('num_feature').agg({
    'ADOM_accuracy_rf': 'mean',
    'ADOM_accuracy_rf_CV': 'mean'
}).reset_index()

# Сортируем по количеству признаков
rf_grouped = rf_grouped.sort_values('num_feature')

# Зависимость точности RF ADOM от числа признаков
axes2[1].plot(rf_grouped['num_feature'], rf_grouped['ADOM_accuracy_rf'], 
              'b-', linewidth=2, label='RF')


axes2[1].set_xlabel('К-во дескрипторов в модели')
axes2[1].set_ylabel('Средняя точность определения надшламовых вод')
#axes2[1].set_title('RF: Гистограмма средней точности и зависимость точности ADOM от числа признаков')
axes2[0].legend(bbox_to_anchor=(1.05, 0.8), loc='lower center')
axes2[1].legend(bbox_to_anchor=(1.05, 0.8), loc='lower center')
axes2[0].grid(False, alpha=0)
axes2[1].grid(False, alpha=0)
plt.tight_layout()
plt.show()

