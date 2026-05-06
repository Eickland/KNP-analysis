import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams["legend.fontsize"] = 17

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import INIT_DATA, CACHE_DATA_PATH, SPECTRA_PATH
from typing import List, Optional, Union

def plot_uv_spectra(
    dfs: List[pd.DataFrame],
    group_by: str = 'class',  # 'class', 'subclass', или None (все спектры по отдельности)
    average: bool = True,
    figsize: tuple = (10, 6),
    xlabel: str = 'Длина волны (нм)',
    ylabel: str = 'Оптическая плотность',
    title: Optional[str] = None,
    show_individual: bool = False,  # Показать индивидуальные спектры полупрозрачными
    individual_alpha: float = 0.3,
    colormap: str = 'tab10',
    legend_loc: str = 'best',
    show_std_spectrum: bool = False,
    fontsize_labels: int = 20,      # Размер шрифта подписей осей
    fontsize_ticks: int = 20,       # Размер шрифта делений на осях
    fontsize_title: int = 24,       # Размер шрифта заголовка
    fontsize_legend: int = 17,      # Размер шрифта легенды    
    **plot_kwargs
):
    """
    Визуализация УФ-спектров с группировкой и усреднением.
    
    Parameters:
    -----------
    dfs : List[pd.DataFrame]
        Список DataFrame с индексами (длины волн) и одним столбцом (оптическая плотность)
    group_by : str
        Уровень группировки: 'class', 'subclass' или None (без группировки)
    average : bool
        Усреднять ли спектры внутри групп (иначе отображать все)
    figsize : tuple
        Размер фигуры
    xlabel, ylabel : str
        Подписи осей
    title : str or None
        Заголовок графика
    show_individual : bool
        Показать ли индивидуальные спектры (полупрозрачными) вместе со средним
    individual_alpha : float
        Прозрачность индивидуальных спектров
    colormap : str
        Название colormap для группировки
    legend_loc : str
        Расположение легенды
    **plot_kwargs
        Дополнительные параметры для plt.plot
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Если group_by = None, отображаем все спектры индивидуально
    if group_by is None:
        for i, df in enumerate(dfs):
            wavelengths = df.index.values
            absorbance = df.iloc[:, 0].values
            label = f"Спектр {i+1}"
            if hasattr(df, 'attrs'):
                if 'class' in df.attrs:
                    label += f" ({df.attrs['class']})"
            ax.plot(wavelengths, absorbance, alpha=individual_alpha, label=label)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        ax.set_ylabel(ylabel, fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)        
        ax.legend(loc=legend_loc, fontsize=fontsize_legend)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "УФ-спектры (индивидуальные)", 
                 fontsize=fontsize_title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax
    
    # Группировка по выбранному атрибуту
    grouped_data = {}
    
    for df in dfs:
        if not hasattr(df, 'attrs'):
            print(f"Предупреждение: DataFrame не имеет атрибутов attrs, пропускаем")
            continue
            
        if group_by not in df.attrs:
            print(f"Предупреждение: В attrs нет ключа '{group_by}', пропускаем")
            continue
            
        group_key = df.attrs[group_by]
        
        if group_key not in grouped_data:
            grouped_data[group_key] = []
        
        grouped_data[group_key].append(df)
    
    if not grouped_data:
        print("Нет данных для отображения")
        return fig, ax
    
    # Получаем уникальные группы и присваиваем цвета
    groups = list(grouped_data.keys())
    cmap = plt.get_cmap(colormap)
    colors = {group: cmap(i / len(groups)) for i, group in enumerate(groups)}
    
    # Для каждой группы усредняем или отображаем все спектры
    for group, group_dfs in grouped_data.items():
        # Получаем общие длины волн (объединение индексов всех спектров в группе)
        all_wavelengths = sorted(set().union(*[set(df.index) for df in group_dfs]))
        
        # Интерполируем все спектры на общую сетку длин волн
        interpolated_spectra = []
        for df in group_dfs:
            wavelengths = df.index.values
            absorbance = df.iloc[:, 0].values
            
            # Интерполяция на общую сетку
            interpolated = np.interp(all_wavelengths, wavelengths, absorbance)
            interpolated_spectra.append(interpolated)
        
        interpolated_spectra = np.array(interpolated_spectra)
        
        if average or show_individual:
            # Вычисляем средний спектр и стандартное отклонение
            mean_spectrum = np.mean(interpolated_spectra, axis=0)
            std_spectrum = np.std(interpolated_spectra, axis=0)
            
            # Отображаем индивидуальные спектры (если нужно)
            if show_individual and len(interpolated_spectra) > 0:
                for i, spectrum in enumerate(interpolated_spectra):
                    ax.plot(all_wavelengths, spectrum, 
                           color=colors[group], alpha=individual_alpha, 
                           linewidth=0.8, label='_nolegend_')
            
            # Отображаем средний спектр
            line = ax.plot(all_wavelengths, mean_spectrum, 
                          color=colors[group], linewidth=2, 
                          label=f"{group} (n={len(group_dfs)})",
                          **plot_kwargs)
            
            # Добавляем область стандартного отклонения
            if show_std_spectrum:
                ax.fill_between(all_wavelengths, 
                            mean_spectrum - std_spectrum, 
                            mean_spectrum + std_spectrum,
                            color=colors[group], alpha=0.2)
        
        else:
            # Отображаем все спектры без усреднения
            for i, df in enumerate(group_dfs):
                wavelengths = df.index.values
                absorbance = df.iloc[:, 0].values
                label = f"{group} (спектр {i+1})" if i == 0 else "_nolegend_"
                ax.plot(wavelengths, absorbance, color=colors[group], 
                       alpha=individual_alpha, label=label)
    
    # Настройка графика
    ax.set_xlabel(xlabel, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax.tick_params(axis='both', labelsize=fontsize_ticks)        
    ax.legend(loc=legend_loc, fontsize=fontsize_legend)
    
    if title is None:
        title = f"УФ-спектры с группировкой по '{group_by}'"
        if average:
            title += " (усредненные)"
    ax.set_title(title, fontsize=fontsize_title)
    
    ax.grid(False, alpha=0.3)
    ax.set_xlim([min(all_wavelengths), max(all_wavelengths)]) # type: ignore
    
    plt.tight_layout()
    return fig, ax

uv_path = Path.joinpath(INIT_DATA)
fluo_path = Path.joinpath(INIT_DATA, "Fluorescence")
fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)

print(len(fluo_spectra))
uv_spectra = ut.load_spectra_data(Path.joinpath(uv_path, "UV","absorption"), uv.read_csv_uv)

fig,ax = plt.subplots(1,1,dpi=300,figsize=(10,8))

i=0

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
sample_list = collection.samples


name_list = ['ADOM-SL1-7-2025','ADOM-SL2-8-2025','ADOM-SL3-8-2025','ADOM-SL4-2-2025','ADOM-SL5-2-2025'
             ,'ADOM-SL6-9-2025','ADOM-SL7-2-2025','ADOM-SL9-1-2025','ADOM-SL10-1-2025','ADOM-SLB-B-1-2025']
name_list = ['LG-11-2-30-2024','LST-S-SBP-2024','PHA-T598-2024','SFA-PW04-2024','CHA-GL02-2024']

uv_list = []

for name,sample in sample_list.items():
    
        uv_df = sample.uv_vis_absorption_smooth
        
        if '2025' not in name:
            continue
        
        if uv_df is None:
            
            print(name, 'uv_df is none')
            continue
        
        uv_df.attrs['class'] = sample.sample_class
        uv_df.attrs['subclass'] = sample.sample_subclass
        uv_df.attrs['TOC'] = sample.org_carbon
        
        uv_df = uv_df.loc[270:]
        
        if uv_df.attrs['TOC'] is None:
            
            print(name, 'TOC - nan')
            continue     
            
        uv_list.append(uv_df/uv_df.attrs['TOC'])

for sample_name in sample_list.keys():

    if sample_name in name_list:
        
        name = sample_name
        short_name = name[:-5]
        
        if name == 'ADOM-SLB-B-1-2025':
            short_name = 'ADOM-B'
        print(sample_name)
        uv_df = sample_list[sample_name].uv_vis_absorption
        if uv_df is None:
            raise ValueError()    
        uv_df.attrs['name'] = short_name
        uv_df.attrs['TOC'] = sample_list[sample_name].org_carbon
        uv.plot_uv(uv_df,ax=ax,norm_by_TOC=True)
        
    i += 1
       
plt.title('')        
plt.xlim(210,800)
plt.ylim(0,0.1)
plt.legend()
plt.tight_layout()        
plt.close()

fig,ax = plt.subplots(1,1,dpi=300,figsize=(10,8))

for sample_name in sample_list.keys():
    

    if sample_name in name_list:
        
        name = sample_name
        short_name = name[:-5]
        
        if name == 'ADOM-SLB-B-1-2025':
            short_name = 'ADOM-B'
        
        fluo = sample_list[sample_name].fluorescence_eem
        
        if fluo is None:
            raise ValueError()

        fluo.attrs['name'] = short_name
        
        fl.plot_2d(fluo,ex_wave=280,ax=ax,norm=True,raman_spline=True)
        
plt.title('')
plt.ylim(0,1)
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')
plt.tight_layout()         
plt.close()

plot_uv_spectra(uv_list,group_by='subclass', colormap='tab20',title='',ylabel='SUVA, $см^{-1}*мг^{-1}*л$')
plt.show()           
