import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from itertools import combinations
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
from HumSpectra.sample import SampleCollection

# Установка размеров шрифтов для всех элементов
plt.rcParams.update({
    'font.size': 18,                    # Базовый размер шрифта
    'axes.titlesize': 18,               # Заголовок осей
    'axes.labelsize': 18,               # Метки осей
    'xtick.labelsize': 16,              # Метки на оси X
    'ytick.labelsize': 16,              # Метки на оси Y
    'legend.fontsize': 16,              # Легенда
    'figure.titlesize': 20              # Заголовок фигуры
})
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.Utilits.config_paths import STAT_PATH, CACHE_DATA_PATH

BoxPlotEnable = False
KdePlotEnable = False
HistPlotEnable = False
ImPlotEnable = False
CustomImplotEnable = False
ScatterPlotEnable = True
CustomScatterPlotEnable = False

def create_hierarchical_scatter(df, class_col, subclass_col, x_col, y_col,s=70):
    """
    Использует HSV цветовое пространство для лучшего контроля насыщенности
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
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
    
    ax.legend(bbox_to_anchor=(1, 1), loc='upper right')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax

def custom_lmplot(data, x, y, hue, no_trend_classes=None, combined_trend_classes=None, 
                  show_equations=False, no_trend_scatter_kws=None, **kwargs):
    """
    Расширенный lmplot с возможностью:
    - Исключить тренд для определенных классов
    - Объединить тренд для групп классов
    - Показывать уравнения регрессии
    - Классы без тренда отображаются с меньшим размером точек и alpha
    
    Parameters:
    -----------
    data : DataFrame
        Исходные данные
    x, y : str
        Названия столбцов для осей X и Y
    hue : str
        Столбец для группировки
    no_trend_classes : list, optional
        Классы, для которых не строить линию тренда
    combined_trend_classes : dict, optional
        Словарь {новое_название: [список_классов]} для объединения трендов
    show_equations : bool
        Показывать уравнения регрессии на графике
    no_trend_scatter_kws : dict, optional
        Параметры для точек классов без тренда (size, alpha и др.)
    **kwargs : 
        Дополнительные параметры для sns.lmplot
    """
    
    if no_trend_classes is None:
        no_trend_classes = []
    
    if combined_trend_classes is None:
        combined_trend_classes = {}
    
    # Параметры по умолчанию для точек без тренда
    default_no_trend_kws = {'s': 20, 'alpha': 0.4}
    if no_trend_scatter_kws:
        default_no_trend_kws.update(no_trend_scatter_kws)
    
    # Создаем копию данных для модификаций
    plot_data = data.copy()
    
    # Добавляем колонку для определения стиля точек
    plot_data['point_style'] = 'normal'
    for cls in no_trend_classes:
        plot_data.loc[plot_data[hue] == cls, 'point_style'] = 'no_trend'
    
    # Создаем отдельную колонку для группировки трендов
    plot_data['trend_group'] = plot_data[hue]
    
    # Применяем объединение трендов
    for new_name, classes in combined_trend_classes.items():
        for class_name in classes:
            plot_data.loc[plot_data[hue] == class_name, 'trend_group'] = new_name
    
    # Создаем основной график без трендов
    fig = sns.lmplot(data=plot_data, x=x, y=y, hue=hue, fit_reg=False,legend=False, **kwargs)
    
    # Перерисовываем точки с разными стилями
    ax = fig.axes[0, 0]
    
    # Удаляем исходные точки (чтобы перерисовать с разными стилями)
    for collection in ax.collections:
        collection.remove()
    
    # Разделяем данные на группы для разного оформления
    normal_data = plot_data[plot_data['point_style'] == 'normal']
    no_trend_data = plot_data[plot_data['point_style'] == 'no_trend']
    
    # Рисуем точки с трендом (нормальные)
    if not normal_data.empty:
        sns.scatterplot(data=normal_data, x=x, y=y, hue=hue, 
                       ax=ax, **{k: v for k, v in kwargs.items() if k != 'scatter_kws'},legend=True)
    
    # Рисуем точки без тренда (уменьшенные)
    if not no_trend_data.empty:
        # Получаем палитру для сохранения цветов
        hue_order = kwargs.get('hue_order', plot_data[hue].unique())
        palette = kwargs.get('palette', sns.color_palette())
        
        # Создаем словарь цветов для классов
        color_dict = {}
        for i, hue_val in enumerate(hue_order):
            color_dict[hue_val] = palette[4+i % len(palette)]
        
        # Рисуем каждую группу без тренда отдельно
        for hue_val in no_trend_data[hue].unique():
            class_data = no_trend_data[no_trend_data[hue] == hue_val]
            ax.scatter(class_data[x], class_data[y], 
                      c=[color_dict[hue_val]], 
                      s=default_no_trend_kws['s'],
                      alpha=default_no_trend_kws['alpha'],
                      label=f'{hue_val} (no trend)')
    
    # Словарь для хранения уравнений регрессии
    equations = {}
    
    # Строим индивидуальные тренды для каждого trend_group
    unique_trend_groups = plot_data['trend_group'].unique()
    
    for trend_group in unique_trend_groups:
        group_data = plot_data[plot_data['trend_group'] == trend_group]
        
        # Проверяем, нужно ли строить тренд для этой группы
        original_classes = plot_data[plot_data['trend_group'] == trend_group][hue].unique()
        should_plot_trend = not any(cls in no_trend_classes for cls in original_classes)
        
        if should_plot_trend and len(group_data) > 1:
            # Вычисляем регрессию
            X = group_data[x].values.reshape(-1, 1)
            y_vals = group_data[y].values
            
            model = LinearRegression()
            model.fit(X, y_vals)
            
            # Генерируем точки для линии тренда
            x_range = np.linspace(X.min(), X.max(), 100)
            y_trend = model.predict(x_range.reshape(-1, 1))
            
            # Определяем цвет для линии тренда
            if trend_group in combined_trend_classes.values():
                # Для объединенных трендов используем нейтральный цвет
                color = 'gray'
            else:
                # Находим индекс цвета для этого класса
                hue_order = kwargs.get('hue_order', plot_data[hue].unique())
                color_idx = list(hue_order).index(original_classes[0])
                palette = kwargs.get('palette', sns.color_palette())
                color = palette[color_idx % len(palette)]
            
            # Рисуем линию тренда
            line = ax.plot(x_range, y_trend, color=color, linestyle='-', 
                          linewidth=2.5, alpha=0.8, 
                          label=f'{trend_group} trend' if trend_group not in original_classes else None)
            
            # Сохраняем уравнение
            slope = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(X, y_vals)
            equations[trend_group] = (slope, intercept, r_squared)
    
    # Добавляем уравнения на график
    if show_equations and equations:
        text_y_pos = 0.95
        text_x_pos = 0.02
        
        for i, (group, (slope, intercept, r_squared)) in enumerate(equations.items()):
            equation_text = f"{group}: y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.2f})"
            ax.text(text_x_pos, text_y_pos - i*0.05, equation_text,
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Обновляем легенду
    ax.legend()
    
    return fig

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0.5)
data.dropna(inplace=True)

MainData = ut.filter_by_iqr_per_group(data,iqr_param=1.5,target_column='Subclass', filter_columns=["asm_350",'Suva'])
numeric_cols = MainData.select_dtypes(include=['float']).columns
MainData[numeric_cols] = MainData[numeric_cols].map(lambda x: float(f"{x:.4g}"))

MainData.set_index(["Class","Subclass","Sample"],inplace=True)

if BoxPlotEnable:

    for feature in numeric_cols:
            
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=300)
        box = sns.boxplot(data = MainData.reset_index(),
                            hue='Class', y=feature, ax=axes)
        ut.add_median_labels(box)
        
        plt.savefig(
        Path.joinpath(STAT_PATH,'Correlations',
                    'Graphs','boxplot',f'boxplot_{feature}.png')
                )
        plt.close()
             
if KdePlotEnable:
    
    combinations_list = list(combinations(numeric_cols, 2))

    for feature_1, feature_2 in combinations_list:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8))
        sns.kdeplot(data = MainData.reset_index(), hue='Class', y=feature_2,x = feature_1, ax=axes, alpha = 0.5)
        
        plt.savefig(
        Path.joinpath(STAT_PATH,'Correlations',
                    'Graphs','kdeplot',f'kdeplot_{feature_1} vs {feature_2}.png')
                )
        plt.close()

if HistPlotEnable:

    for feature in numeric_cols:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8))
        sns.histplot(data = MainData, hue='Subclass', x=feature, ax=axes,
                        multiple="stack",palette="rainbow",edgecolor=".3",linewidth=.5,log_scale=False,)
        
        plt.savefig(
        Path.joinpath(STAT_PATH,'Correlations',
                    'Graphs','histplot',f'histplot_{feature}.png')
                )
        plt.close()
        
if ImPlotEnable:
    
    combinations_list = list(combinations(numeric_cols, 2))

    for feature_1, feature_2 in combinations_list:
                    
        sns.lmplot(data = MainData.reset_index(), hue='Class', y=feature_1, x = feature_2)
        
        plt.savefig(
        Path.joinpath(STAT_PATH,'Correlations',
                    'Graphs','implot',f'implot_{feature_1} vs {feature_2}.png')
                )
        plt.close()
        
if ScatterPlotEnable:
            
    combinations_list = list(combinations(numeric_cols, 2))

    for feature_1, feature_2 in combinations_list:
        
        fig, axes = plt.subplots(1,1,figsize=(8,8),dpi=100)
        
        sns.scatterplot(data = MainData, hue='Subclass', y=feature_2,x = feature_1, ax=axes, alpha = 1,legend=True)
        """
        plt.legend(
        bbox_to_anchor=(1.05, 1),  # Позиция относительно axes (1,1 - правый верхний угол)
                loc='upper left',          # Якорь легенды
                borderaxespad=0.0          # Отступ от axes
            )
        """    
        plt.tight_layout()
        plt.savefig(
        Path.joinpath(STAT_PATH,'Correlations',
                    'Graphs','scatterplot',f'scatterplot_{feature_1} vs {feature_2}.png')
                )
        plt.close()