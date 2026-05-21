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
                          label=f"{group}",
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
    
    ax.set_xlim([min(all_wavelengths), max(all_wavelengths)]) # type: ignore
    
    plt.tight_layout()
    return fig, ax

def plot_uv_spectra_list(spectra_list: List[pd.DataFrame],
                          norm_by_TOC: bool = False,
                          ylim: Optional[float] = None,
                          uniform_ylim: bool = True,
                          ylim_margin: float = 0.05,
                          dpi:int = 300,
                          numeration_list:list = ['а', 'б', 'в', 'г']) -> None:
    """
    Визуализирует 4 УФ спектра в сетке 2x2 с нумерацией буквами русского алфавита.
    
    :param spectra_list: список из 4 DataFrame с УФ спектрами
    :param norm_by_TOC: нормализовать по TOC
    :param ylim: верхняя граница по оси Y (если указана, переопределяет uniform_ylim)
    :param uniform_ylim: автоматически выровнять пределы по оси Y для всех спектров
    :param ylim_margin: отступ от максимального значения (в долях) при uniform_ylim=True
    """
    
    if len(spectra_list) != 4:
        raise ValueError("Список спектров должен содержать ровно 4 элемента")
    
    # Русский алфавит для нумерации
    russian_letters = numeration_list
    
    # Если uniform_ylim=True и ylim не указан, вычисляем общие пределы
    global_ylim = None
    if uniform_ylim and ylim is None:
        global_max = 0
        global_min = float('inf')
        
        for data in spectra_list:
            data_copy = data.copy()
            
            # Нормализация по TOC для расчета пределов
            if norm_by_TOC:
                if "TOC" not in data_copy.attrs:
                    raise KeyError("В метаданных спектра должно быть значения содержания органического углерода")
                data_copy = data_copy / data_copy.attrs['TOC']
            
            # Находим min и max в данных
            column_name = data_copy.columns[0]
            current_min = data_copy[column_name].min()
            current_max = data_copy[column_name].max()
            
            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)
        
        # Добавляем отступы
        y_range = global_max - global_min
        y_min = max(0, global_min - y_range * ylim_margin)  # Не уходим в отрицательные значения
        y_max = global_max + y_range * ylim_margin
        
        global_ylim = (y_min, y_max)
    
    # Создаем фигуру с сабплотами 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),dpi=dpi)
    axes = axes.flatten()
    
    print("Соответствие букв и спектров:")
    print("-" * 30)
    
    # Для каждого спектра строим график
    for idx, (data, letter) in enumerate(zip(spectra_list, russian_letters)):
        ax = axes[idx]
        
        # Определяем пределы для текущего графика
        if ylim is not None:
            current_ylim = ylim
        elif global_ylim is not None:
            current_ylim = global_ylim[1]  # Передаем только верхнюю границу для совместимости
        else:
            current_ylim = None
        
        # Используем существующую функцию plot_uv
        ax = uv.plot_uv(
            data=data,
            xlabel=True,
            ylabel=True,
            title=False,
            ylim=current_ylim,
            norm_by_TOC=norm_by_TOC,
            ax=ax
        )
        
        # Если используются одинаковые пределы, устанавливаем их явно
        if global_ylim is not None and ylim is None:
            ax.set_ylim(global_ylim)
        
        # Добавляем нумерацию буквой в левом верхнем углу
        ax.text(0.02, 0.98, f'{letter})', 
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Получаем имя спектра для вывода в консоль
        if "name" in data.attrs:
            spectrum_name = data.attrs["name"]
        else:
            spectrum_name = f"spectrum_{idx+1}"
        
        print(f"{letter}) -> {spectrum_name}")
        
        # Убираем легенду, если она появилась
        if ax.get_legend():
            ax.get_legend().remove() # type: ignore
        ax.set_xticks([200,300,400,500,600,700])
    # Выводим информацию о пределах, если используется автовыравнивание
    if global_ylim is not None and ylim is None:
        print("-" * 30)
        print(f"Автоматически установлены общие пределы по оси Y: [{global_ylim[0]:.3f}, {global_ylim[1]:.3f}]")
        print(f"(отступ: {ylim_margin*100}% от размаха данных)")
    
    print("-" * 30)
    
    # Общие настройки
    plt.tight_layout()

def plot_fluorescence_spectra_list(spectra_list: List[pd.DataFrame],
                                    norm_by_TOC: bool = False,
                                    uniform_clim: bool = True,
                                    clim: Optional[Union[float, tuple]] = None,
                                    clim_margin: float = 0.05,
                                    clim_mode: str = 'max_min',
                                    dpi:int = 300,
                                    numeration_list:list = ['а', 'б', 'в', 'г']                                    ) -> None:
    """
    Визуализирует 4 спектра флуоресценции (3D тепловые карты) в сетке 2x2 
    с нумерацией буквами русского алфавита.
    
    :param spectra_list: список из 4 DataFrame с спектрами флуоресценции
    :param norm_by_TOC: нормализовать по TOC (нужен атрибут 'TOC')
    :param uniform_clim: автоматически выровнять цветовые пределы для всех спектров
    :param clim: верхняя граница (float) или кортеж (нижняя, верхняя) для цветовой шкалы
    :param clim_margin: отступ от экстремальных значений (в долях) при uniform_clim=True
    :param clim_mode: режим выравнивания ('max_min', 'max_only', 'percentile')
    """
    
    if len(spectra_list) != 4:
        raise ValueError("Список спектров должен содержать ровно 4 элемента")
    
    # Русский алфавит для нумерации
    russian_letters = numeration_list
    
    # Копируем и нормализуем спектры если нужно
    processed_spectra = []
    for data in spectra_list:
        data_copy = data.copy()
        
        # Нормализация по TOC
        if norm_by_TOC:
            if "TOC" not in data_copy.attrs:
                raise KeyError("В метаданных спектра должно быть значение содержания органического углерода")
            data_copy = data_copy / data_copy.attrs['TOC']
        
        processed_spectra.append(data_copy)
    
    # Обработка параметра clim
    if clim is not None:
        if isinstance(clim, (int, float)):
            clim_tuple = (0, clim)
        elif isinstance(clim, tuple) and len(clim) == 2:
            clim_tuple = clim
        else:
            raise ValueError("clim должен быть числом (верхняя граница) или кортежем (нижняя, верхняя)")
    else:
        clim_tuple = None
    
    # Вычисляем общие цветовые пределы если нужно
    global_clim = None
    if uniform_clim and clim_tuple is None:
        all_values = []
        
        for data in processed_spectra:
            all_values.extend(data.values.flatten())
        
        all_values = np.array(all_values)
        all_values = all_values[~np.isnan(all_values)]  # Убираем NaN
        
        if len(all_values) == 0:
            raise ValueError("Нет валидных данных для определения цветовых пределов")
        
        if clim_mode == 'max_min':
            vmin = all_values.min()
            vmax = all_values.max()
        elif clim_mode == 'max_only':
            vmin = 0
            vmax = all_values.max()
        elif clim_mode == 'percentile':
            vmin = 0
            vmax = np.percentile(all_values, 95)  # 95-й процентиль игнорирует выбросы
        else:
            raise ValueError("clim_mode должен быть 'max_min', 'max_only' или 'percentile'")
        
        # Добавляем отступы
        vrange = vmax - vmin
        vmin_final = max(0, vmin - vrange * clim_margin)
        vmax_final = vmax + vrange * clim_margin
        
        global_clim = (vmin_final, vmax_final)
    
    # Создаем фигуру с сабплотами 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=dpi)
    axes = axes.flatten()
    
    print("Соответствие букв и спектров:")
    print("-" * 40)
    
    # Для каждого спектра строим тепловую карту
    for idx, (data, letter) in enumerate(zip(processed_spectra, russian_letters)):
        ax = axes[idx]
        
        # Используем существующую функцию plot_heat_map
        ax = fl.plot_heat_map(
            data=data,
            ax=ax,
            xlabel=True,
            ylabel=True,
            title=False  # Заголовок отключаем, т.к. используем нумерацию
        )
        
        # Устанавливаем цветовые пределы
        if clim_tuple is not None:
            pcolormesh_obj = ax.collections[0]
            pcolormesh_obj.set_clim(clim_tuple)
        elif global_clim is not None:
            pcolormesh_obj = ax.collections[0]
            pcolormesh_obj.set_clim(global_clim) # type: ignore
        
        # Добавляем нумерацию буквой в левом верхнем углу
        ax.text(0.02, 0.98, f'{letter})', 
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Получаем имя спектра для вывода в консоль
        if "name" in data.attrs:
            spectrum_name = data.attrs["name"]
        else:
            spectrum_name = f"spectrum_{idx+1}"
        
        print(f"{letter}) -> {spectrum_name}")
    
    # Добавляем общую colorbar для всех графиков (опционально)
    if uniform_clim or clim_tuple is not None:
        # Создаем общую colorbar справа от всех графиков
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # type: ignore
        
        if clim_tuple is not None:
            norm = plt.Normalize(clim_tuple[0], clim_tuple[1]) # type: ignore
        elif global_clim is not None:
            norm = plt.Normalize(global_clim[0], global_clim[1]) # type: ignore
        
        # Берем последний pcolormesh для создания colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('rainbow'), norm=norm) # type: ignore
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        if norm_by_TOC:
            cbar.set_label('Интенсивность / TOC, отн. ед.', fontsize=12)
        else:
            cbar.set_label('Интенсивность флуоресценции, отн. ед.', fontsize=12)
    
    # Выводим информацию о цветовых пределах
    if clim_tuple is not None:
        print("-" * 40)
        print(f"Использованы заданные цветовые пределы: [{clim_tuple[0]:.3f}, {clim_tuple[1]:.3f}]")
    elif global_clim is not None:
        print("-" * 40)
        print(f"Автоматически установлены общие цветовые пределы: [{global_clim[0]:.3f}, {global_clim[1]:.3f}]")
        print(f"(режим: {clim_mode}, отступ: {clim_margin*100}% от размаха данных)")
    
    print("-" * 40)
    
    # Общие настройки
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # type: ignore # Оставляем место для colorbar
    
uv_path = Path.joinpath(INIT_DATA)
fluo_path = Path.joinpath(INIT_DATA, "Fluorescence")
fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)

print(len(fluo_spectra))
uv_spectra = ut.load_spectra_data(Path.joinpath(uv_path, "UV","absorption"), uv.read_csv_uv)

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
sample_list = collection.samples


name_list = ['ADOM-SL1-7-2025','ADOM-SL2-8-2025','ADOM-SL3-8-2025','ADOM-SL4-2-2025','ADOM-SL5-2-2025'
             ,'ADOM-SL6-9-2025','ADOM-SL7-2-2025','ADOM-SL9-1-2025','ADOM-SL10-1-2025','ADOM-SLB-B-1-2025']
name_list = ['LG-11-2-30-2024','LST-S-SBP-2024','PHA-T598-2024','SFA-PW04-2024','CHA-GL02-2024']
name_list_choosed = ['ADOM-SL1-7-2025','ADOM-SL2-8-2025','ADOM-SL3-8-2025','ADOM-SL4-2-2025']

uv_list = []
uv_list_choosed = []
fluo_list_choosed = []

#Сборка полного списка уф спектров
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
        
        uv_df = uv_df.loc[260:]
        
        if uv_df.attrs['TOC'] is None:
            
            print(name, 'TOC - nan')
            continue     
            
        uv_list.append(uv_df/uv_df.attrs['TOC'])

#Сборка списка из 4 спектров
for name in name_list_choosed:
    
    sample = sample_list[name]
    uv_df = sample.uv_vis_absorption_smooth
    fluo_df = sample.cutted_fluo_eem
    
    if uv_df is None:
        raise ValueError(f'{name}', 'uv_spectra is empty')
    
    if fluo_df is None:
        raise ValueError(f'{name}', 'fluo_spectra is empty') 
       
    uv_df.attrs['class'] = sample.sample_class
    uv_df.attrs['subclass'] = sample.sample_subclass
    uv_df.attrs['TOC'] = sample.org_carbon
    uv_df = uv_df.loc[:700]
    
    fluo_df.attrs['class'] = sample.sample_class
    fluo_df.attrs['subclass'] = sample.sample_subclass
    fluo_df.attrs['TOC'] = sample.org_carbon    
    
    if uv_df.attrs['TOC'] is None:
        raise ValueError(f'{name}', 'TOC is none')
    
    uv_list_choosed.append(uv_df)
    fluo_list_choosed.append(fluo_df)

ModeAverageSpectra = True
ModeFourSpectra = False
FluoFourSpectra = False
    
if ModeAverageSpectra:
    
    plot_uv_spectra(uv_list,group_by='subclass', colormap='tab20',title='',ylabel='SUVA, $см^{-1}*мг^{-1}*л$')
    plt.savefig(r"D:\lab\KNP-analysis\Received_data\Spectra\UV_plots\Average_by_subclas_uv_spectra.png")
    plt.close()           

if ModeFourSpectra:

    plot_uv_spectra_list(uv_list_choosed,norm_by_TOC=True,numeration_list=['а','б','в','г'], ylim=0.027)
    plt.savefig(r"D:\lab\KNP-analysis\Received_data\Spectra\UV_plots\Article_format_four_uv_spectra.png")
    plt.close()
    
if FluoFourSpectra:
    plot_fluorescence_spectra_list(fluo_list_choosed,numeration_list=['и','e','ж','з'],clim=3)
    plt.savefig(r"D:\lab\KNP-analysis\Received_data\Spectra\Fluo_plots\Article_format_four_fluo_spectra.png")
    plt.close()