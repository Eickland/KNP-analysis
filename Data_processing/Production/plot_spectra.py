from pathlib import Path
import matplotlib.pyplot as plt
import sys
from typing import List, Any
import pandas as pd
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
import random
project_root = Path.cwd()
sys.path.insert(0, str(project_root))

FluoPlotEnable = False
UVPlotEnable = False

plt.rcParams['figure.dpi'] = 300

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

date_list = ["2024"]
date_folder_list = [INIT_DATA_2024]

def split_dataframes_by_class(dataframes: List[pd.DataFrame], type='subclass') -> List[List[pd.DataFrame]]:
    """
    Разделяет список DataFrame по классам, указанным в df.attrs['class']
    
    Args:
        dataframes: Список DataFrame для разделения
        
    Returns:
        List[List[pd.DataFrame]]: Список списков DataFrame, сгруппированных по классам
    """
    # Собираем все уникальные классы
    all_classes = set()
    
    for df in dataframes:
        if hasattr(df, 'attrs') and type in df.attrs:
            class_value = df.attrs[type]
            # Если класс - это список, добавляем все элементы
            if isinstance(class_value, list):
                all_classes.update(class_value)
            else:
                all_classes.add(class_value)
    
    # Создаем словарь для группировки
    class_groups = {cls: [] for cls in all_classes}
    
    # Распределяем DataFrame по классам
    for df in dataframes:
        if hasattr(df, 'attrs') and type in df.attrs:
            class_value = df.attrs[type]
            
            if isinstance(class_value, list):
                # Если несколько классов, добавляем DataFrame во все соответствующие группы
                for cls in class_value:
                    if cls in class_groups:
                        class_groups[cls].append(df)
            else:
                # Если один класс, добавляем в соответствующую группу
                if class_value in class_groups:
                    class_groups[class_value].append(df)
    
    # Преобразуем словарь в список списков
    return list(class_groups.values())

def add_toc_to_spectra(spectra: List[pd.DataFrame], 
                               toc_table: pd.DataFrame, 
                               name_col: str = 'Sample', 
                               toc_col: str = 'C_org',
                               verbose: bool = True) -> List[pd.DataFrame]:
    """
    Добавляет параметр 'TOC' в attrs каждого спектра и удаляет спектры без TOC
    
    Args:
        spectra: Список DataFrame спектров
        toc_table: DataFrame с таблицей соответствия имен и значений TOC
        name_col: Название столбца с именами в toc_table (по умолчанию 'Sample')
        toc_col: Название столбца со значениями TOC в toc_table (по умолчанию 'C_org')
        verbose: Выводить ли информацию об удаленных спектрах
        
    Returns:
        List[pd.DataFrame]: Список спектров с добавленным параметром 'TOC' в attrs
                            (спектры без TOC удалены)
    """
    
    # Создаем словарь для быстрого поиска TOC по имени
    toc_dict = dict(zip(toc_table[name_col], toc_table[toc_col]))
    
    updated_spectra = []
    removed_count = 0
    
    for spectrum in spectra:
        # Создаем копию DataFrame
        spectrum_copy = spectrum.copy()
        
        # Проверяем наличие атрибута 'name'
        if hasattr(spectrum, 'attrs') and 'name' in spectrum.attrs:
            spectrum_name = spectrum.attrs['name']
            
            # Ищем соответствие в таблице TOC
            if spectrum_name in toc_dict:
                # Создаем копию attrs и добавляем TOC
                new_attrs = spectrum.attrs.copy()
                new_attrs['TOC'] = toc_dict[spectrum_name]
                spectrum_copy.attrs = new_attrs
                updated_spectra.append(spectrum_copy)
            else:
                removed_count += 1
                if verbose:
                    print(f"Удален спектр: '{spectrum_name}' - TOC не найден")
        else:
            removed_count += 1
            if verbose:
                print(f"Удален спектр: отсутствует атрибут 'name'")
    
    if verbose:
        print(f"Обработка завершена. Удалено спектров: {removed_count}, осталось: {len(updated_spectra)}")
    
    return updated_spectra

carbon_data = pd.read_excel(Path.joinpath(INIT_DATA_2024, 'C_org.xlsx'))
carbon_data.Sample = carbon_data.Sample.apply(lambda x: x+'-2024')

for i, folder in enumerate(date_folder_list):
    
    fluo_path = Path.joinpath(folder, "Fluorescence")
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")
    
    fluo_spectra = ut.load_spectra_data(fluo_path, fl.read_fluo_3d)
    uv_abs_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
    
    uv_toc_spectra = add_toc_to_spectra(uv_abs_spectra,carbon_data)
    
    uv_class_spectra = split_dataframes_by_class(uv_toc_spectra,type='class')
    fluo_subclass_spectra = split_dataframes_by_class(fluo_spectra,type= 'class')
    
    if i == 1:
        uv_refl_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")
    fig,ax = plt.subplots(1,1,figsize=(7,7),dpi=300)
    for fluo_subclass in fluo_subclass_spectra: 
        random_element = random.sample(fluo_subclass, 1)[0]
        if random_element.attrs['subclass'] in ['Lg','Lst','SHA','SFA','SHF','PFA','PHF','PHA','CHF','CHA']:
        
            fl_spectra = fl.cut_spectra(random_element,255,450,255,600)
            fl_spectra = ut.normilize_by_max(fl_spectra)
            fl_spectra_280 = fl.cut_raman_spline(em_wvs=fl_spectra.index.to_numpy(),fl=fl_spectra[280].to_numpy(),exc_wv=280)
            ax.plot(fl_spectra.index.to_numpy(),fl_spectra_280,label=random_element.attrs['name'])
            
            ax.set_title('Спектр флуоресценции при 280 нм длины волны возбуждения')
            ax.set_ylabel('Интенсивность')
            ax.set_xlabel('Длина волны испускания, нм')
            ax.set_ylim((0.0,1.1))
        else:
            continue
    plt.legend()
    plt.tight_layout()
    plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","Fluorescence","280_nm",f"Карты+байкал")
                        )
    plt.close()
    
    if FluoPlotEnable:
    
        for fl_spectra in fluo_spectra:
            
            fl_spectra = fl.cut_spectra(fl_spectra,255,450,255,600)
            fl.plot_heat_map(fl_spectra)
            
            plt.tight_layout()
            plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","Fluorescence","Individual_spectra",f"{fl_spectra.attrs['name']}")
                        )
            plt.close()
        
    fig,ax = plt.subplots(1,1,figsize=(7,7),dpi=300) 
    for uv_class_list in uv_class_spectra:
        
        random_elements = random.sample(uv_class_list, 1)
        if random_elements[0].attrs['class'] in ['ADOM','Baikal']:
            
            continue
        else:
            for uv_spectra in random_elements:
                
                uv_spectra = uv_spectra.iloc[20:]
                
                uv.plot_uv(uv_spectra,norm_by_TOC=True,name=uv_spectra.attrs['name'],ax=ax,title=False)
                ax.set_title('Нормированный спектр поглощения')
                ax.set_xlabel('Длина волны, нм')
    plt.legend()    
    plt.tight_layout()  
    plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Grouped_spectra",f"Нормированный спектр поглощения_2") # type: ignore
                        )
    plt.close()
    
    if UVPlotEnable:
             
        for uv_spectra in uv_abs_spectra:
            
            uv_spectra = uv_spectra.iloc[20:]
            
            uv.plot_uv(uv_spectra)
            
            plt.tight_layout()
            
            plt.savefig(
                Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Individual_spectra",f"{uv_spectra.attrs['name']}")
                        )
            plt.close()
                
        
        if i == 1:
            
            for uv_spectra in uv_refl_spectra: # type: ignore
                
                uv.plot_uv(uv_spectra)
                
                plt.tight_layout()
                plt.savefig(
                    Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Reflection","Individual_spectra",f"{uv_spectra.attrs['name']}")
                            )
                plt.close()
        