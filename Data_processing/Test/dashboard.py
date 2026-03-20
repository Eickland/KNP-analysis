import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.Utilits.config_paths import INIT_DATA_2024, INIT_DATA_2025, STAT_PATH, PARAFAC_PATH

def plot_dataframe_dashboard(df, figsize=(20, 15), cmap='coolwarm'):
    """
    Рисует дашборд для DataFrame с мультииндексом ('Class','Subclass','Sample')
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с мультииндексом ('Class','Subclass','Sample') в строках
        и дескрипторами в столбцах
    figsize : tuple, default=(20, 15)
        Размер фигуры
    cmap : str, default='coolwarm'
        Цветовая схема для тепловой карты
    """
    
    # Создаем фигуру с GridSpec для лучшего контроля над расположением
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.2])
    
    # 1. Картинка: количество образцов (общее, по классам, по подклассам)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Подсчет количества образцов
    total_samples = len(df)
    class_counts = df.index.get_level_values('Class').value_counts()
    subclass_counts = df.index.get_level_values('Subclass').value_counts()
    
    # Создаем bar plot для классов
    y_pos = np.arange(len(class_counts))
    ax1.barh(y_pos, class_counts.values, alpha=0.7, color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(class_counts.index)
    ax1.set_xlabel('Количество образцов')
    ax1.set_title(f'Количество образцов по классам\n(Всего: {total_samples} образцов)')
    ax1.grid(axis='x', alpha=0.3)
    
    # Добавляем аннотации с количеством
    for i, v in enumerate(class_counts.values):
        ax1.text(v + 0.1, i, str(v), va='center')
    
    # 2. Картинка: количество по подклассам
    ax2 = fig.add_subplot(gs[0, 1])
    
    y_pos2 = np.arange(len(subclass_counts))
    colors2 = plt.get_cmap('Set3')(np.linspace(0, 1, len(subclass_counts)))
    ax2.barh(y_pos2, subclass_counts.values, alpha=0.7, color=colors2)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(subclass_counts.index)
    ax2.set_xlabel('Количество образцов')
    ax2.set_title('Количество образцов по подклассам')
    ax2.grid(axis='x', alpha=0.3)
    
    # Добавляем аннотации с количеством
    for i, v in enumerate(subclass_counts.values):
        ax2.text(v + 0.1, i, str(v), va='center')
    
    # 3. Картинка: Boxplot по классам
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Подготовка данных для boxplot по классам
    class_data = []
    class_labels = []
    
    for class_name in df.index.get_level_values('Class').unique():
        class_df = df.xs(class_name, level='Class')
        # Берем средние значения по всем дескрипторам для каждого образца
        sample_means = class_df.mean(axis=1)
        class_data.append(sample_means.values)
        class_labels.append(class_name)
    
    ax3.boxplot(class_data, label=class_labels)
    ax3.set_ylabel('Средние значения дескрипторов')
    ax3.set_title('Распределение средних значений по классам')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Картинка: Boxplot по подклассам
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Подготовка данных для boxplot по подклассам
    subclass_data = []
    subclass_labels = []
    
    for subclass_name in df.index.get_level_values('Subclass').unique():
        try:
            subclass_df = df.xs(subclass_name, level='Subclass')
            # Берем средние значения по всем дескрипторам для каждого образца
            sample_means = subclass_df.mean(axis=1)
            subclass_data.append(sample_means.values)
            subclass_labels.append(subclass_name)
        except KeyError:
            continue
    
    ax4.boxplot(subclass_data, label=subclass_labels)
    ax4.set_ylabel('Средние значения дескрипторов')
    ax4.set_title('Распределение средних значений по подклассам')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Картинка: Тепловая карта корреляций
    ax5 = fig.add_subplot(gs[2, :])
    
    # Вычисляем матрицу корреляций
    corr_matrix = df.corr()
    
    # Рисуем тепловую карту
    im = ax5.imshow(corr_matrix.values, cmap=cmap, aspect='auto')
    
    # Устанавливаем метки
    ax5.set_xticks(np.arange(len(corr_matrix.columns)))
    ax5.set_yticks(np.arange(len(corr_matrix.columns)))
    ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
    ax5.set_yticklabels(corr_matrix.columns)
    
    # Добавляем цветовую шкалу
    cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
    cbar.set_label('Коэффициент корреляции')
    
    ax5.set_title('Тепловая карта корреляций между дескрипторами')
    
    # Добавляем значения корреляций в ячейки
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительная текстовая информация
    print("="*50)
    print("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("="*50)
    print(f"Общее количество образцов: {total_samples}")
    print(f"Количество классов: {len(class_counts)}")
    print(f"Количество подклассов: {len(subclass_counts)}")
    print(f"Количество дескрипторов: {len(df.columns)}")
    print("\nРаспределение по классам:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} образцов")
    return fig

fluo_path = Path.joinpath(INIT_DATA_2024, "Fluorescence")
uv_path = Path.joinpath(INIT_DATA_2024, "UV","Individual_spectra")

carbon_data = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'C_org.xlsx'))
carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+'-'+'2024')

geodata = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'GeoData.xlsx'))
geodata.Sample = geodata.Sample.apply(lambda x: x+'-'+'2024')

fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")

uv_spectra_f = []

for spectra in uv_spectra:
    
    if float(spectra.min().iloc[0]) <= 0:
        
        spectra = spectra - spectra.min() + 0.00001
    uv_spectra_f.append(spectra)
    
fl_data = ut.spectra_to_df(fluo_spectra, {
    "Asm_350": fl.asm_350, "Asm_280": fl.asm_280,
    "FIX": fl.fluo_index, "HIX": fl.humin_index
}, class_filter=None)

uv_data = ut.spectra_to_df(uv_spectra_f, {
    "E2E3": uv.e2_e3, "epsilon": uv.single_density
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


MainData = ut.filter_by_iqr_per_group(MainData.reset_index(),'Class',iqr_param=1.5,
                                        filter_columns=['Asm_350',])


MainData.set_index(["Class","Subclass","Sample"],inplace=True)
MainData.drop(columns=['epsilon'],inplace=True)
plot_dataframe_dashboard(MainData)
plt.tight_layout()