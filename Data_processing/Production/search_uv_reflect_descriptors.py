import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from itertools import combinations
from scipy import integrate

import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2025, STAT_PATH

uv_path = Path.joinpath(INIT_DATA_2025, "UV","Individual_spectra")
uv_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")

filtred_uv_spectra = [spectra for spectra in uv_spectra 
                      if spectra.attrs['subclass'] in ['Baikal','Storage 2','Storage 10', 'Storage 4', 'Storage 7']]

def analyze_multiple_spectra_ratios(spectra_list, n_segments=12, cv_folds=5):
    """
    Анализирует классификацию для нескольких спектров
    
    Parameters:
    -----------
    spectra_list : list
        Список DataFrame спектров, каждый с атрибутом 'subclass'
    n_segments : int
        Количество участков для разделения спектра
    cv_folds : int
        Количество фолдов для кросс-валидации
    
    Returns:
    --------
    pandas.DataFrame
        Таблица с результатами классификации
    """
    
    all_integrals = []
    all_subclasses = []
    segment_info = None
    
    # Обрабатываем каждый спектр
    for spectrum_df in spectra_list:
        if not hasattr(spectrum_df, 'attrs') or 'subclass' not in spectrum_df.attrs:
            continue
            
        wavelengths = spectrum_df.index
        reflectance = spectrum_df.iloc[:, 0]
        
        # Создаем границы участков (только для первого спектра)
        if segment_info is None:
            segment_bounds = np.linspace(wavelengths.min(), wavelengths.max(), n_segments + 1)
            segment_info = []
            for i in range(n_segments):
                lower_bound = segment_bounds[i]
                upper_bound = segment_bounds[i + 1]
                segment_info.append(f"{lower_bound:.1f}-{upper_bound:.1f}")
        
        # Вычисляем интегралы для каждого участка
        integrals = []
        for i in range(n_segments):
            lower_bound = segment_bounds[i]
            upper_bound = segment_bounds[i + 1]
            
            mask = (wavelengths >= lower_bound) & (wavelengths <= upper_bound)
            segment_wavelengths = wavelengths[mask]
            segment_reflectance = reflectance[mask]
            
            if len(segment_wavelengths) > 1:
                integral = np.trapezoid(segment_reflectance, segment_wavelengths)
            else:
                integral = 0
            
            integrals.append(integral)
        
        all_integrals.append(integrals)
        all_subclasses.append(spectrum_df.attrs['subclass'])
    
    # Преобразуем в массивы
    X_integrals = np.array(all_integrals)
    y = np.array(all_subclasses)
    
    # Находим все уникальные пары участков
    pairs = list(combinations(range(n_segments), 2))
    
    # Анализируем каждую пару
    results = []
    
    for i, j in pairs:
        # Вычисляем отношения для всех спектров
        ratios = []
        for integrals in all_integrals:
            if integrals[j] != 0:
                ratio = integrals[i] / integrals[j]
            else:
                ratio = 0
            ratios.append(ratio)
        
        X_ratio = np.array(ratios).reshape(-1, 1)
        
        # Оцениваем классификатор с кросс-валидацией
        try:
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(lda, X_ratio, y, cv=min(cv_folds, len(y)))
            mean_accuracy = np.mean(scores)
            
            results.append({
                'segment_pair': f"{segment_info[i]} / {segment_info[j]}",
                'segments_indices': f"{i+1}/{j+1}",
                'accuracy': mean_accuracy,
                'std_accuracy': np.std(scores)
            })
            
        except Exception as e:
            print(f"Ошибка для пары {i}/{j}: {e}")
    
    return pd.DataFrame(results).sort_values('accuracy', ascending=False)

num_segment = 40

res = analyze_multiple_spectra_ratios(filtred_uv_spectra,n_segments=num_segment)
res.to_excel(Path.joinpath(STAT_PATH,f'Статистика дескрипторов зеркального отражения, n={num_segment}.xlsx'))