from pathlib import Path
import matplotlib.pyplot as plt
import sys

import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline, interp1d
from scipy.ndimage import uniform_filter1d

def smooth_uv_spectrum_with_interpolation(df, window_size=3, threshold_factor=2.0, 
                                        interpolation_method='cubic', interpolate_whole_spectrum=False):
    """
    Находит и сглаживает участки с резкими перепадами интенсивности в УФ спектре
    с использованием интерполяции
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с колонками 'nm' (длина волны) и 'absorption' (интенсивность)
    window_size : int
        Размер окна для обнаружения перепадов (по умолчанию 5)
    threshold_factor : float
        Коэффициент для определения порога перепадов (по умолчанию 2.0)
    interpolation_method : str
        Метод интерполяции: 'cubic', 'linear', 'polynomial' (по умолчанию 'cubic')
    interpolate_whole_spectrum : bool
        Если True - интерполирует весь спектр, если False - только проблемные участки
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame с исходными данными и дополнительными колонками:
        - 'absorption_smooth': сглаженный спектр
        - 'is_anomaly': метки аномальных точек
    """
    
    # Создаем копию DataFrame
    result_df = df.copy()
    wavelengths = df.index.values
    absorption = df[df.columns[0]].values
    
    # 1. Вычисляем производные для обнаружения резких изменений
    gradient = np.gradient(absorption)
    second_derivative = np.gradient(gradient)
    
    # 2. Находим аномальные точки по первой и второй производной
    grad_threshold = threshold_factor * np.std(gradient)
    second_grad_threshold = threshold_factor * np.std(second_derivative)
    
    # Точки с резкими изменениями интенсивности
    sharp_grad_indices = np.where(np.abs(gradient) > grad_threshold)[0]
    sharp_second_grad_indices = np.where(np.abs(second_derivative) > second_grad_threshold)[0]
    
    # 3. Объединяем все проблемные точки
    problematic_indices = np.unique(np.concatenate([
        sharp_grad_indices, 
        sharp_second_grad_indices
    ]))
    
    # Создаем маску аномальных точек
    anomaly_mask = np.zeros(len(absorption), dtype=bool)
    
    if interpolate_whole_spectrum:
        # 4. Вариант 1: Интерполяция всего спектра
        if interpolation_method == 'cubic':
            interpolator = CubicSpline(wavelengths, absorption)
        elif interpolation_method == 'linear':
            interpolator = interp1d(wavelengths, absorption, kind='linear', 
                                  fill_value='extrapolate') # type: ignore
        else:  # polynomial
            interpolator = interp1d(wavelengths, absorption, kind='quadratic', 
                                  fill_value='extrapolate') # type: ignore
        
        # Создаем более плотную сетку для интерполяции
        dense_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), 
                                      len(wavelengths) * 2)
        dense_absorption = interpolator(dense_wavelengths)
        
        # Возвращаем к исходной сетке
        final_interpolator = interp1d(dense_wavelengths, dense_absorption, 
                                    kind='linear', bounds_error=False, 
                                    fill_value='extrapolate') # type: ignore
        smoothed = final_interpolator(wavelengths)
        
    else:
        # 5. Вариант 2: Интерполяция только проблемных участков
        smoothed = absorption.copy()
        
        if len(problematic_indices) > 0:
            # Расширяем области вокруг проблемных точек
            for idx in problematic_indices:
                start = max(0, idx - window_size)
                end = min(len(absorption), idx + window_size + 1)
                anomaly_mask[start:end] = True
            
            # Находим непрерывные проблемные сегменты
            segments = []
            current_segment = []
            
            for i in range(len(anomaly_mask)):
                if anomaly_mask[i]:
                    current_segment.append(i)
                elif current_segment:
                    segments.append(current_segment)
                    current_segment = []
            if current_segment:
                segments.append(current_segment)
            
            # Интерполируем каждый проблемный сегмент
            for segment in segments:
                if len(segment) < 3:  # Слишком короткий сегмент - пропускаем
                    continue
                    
                start_idx = max(0, segment[0] - 1)
                end_idx = min(len(absorption) - 1, segment[-1] + 1)
                
                # Используем точки до и после сегмента для интерполяции
                left_idx = max(0, start_idx - 2)
                right_idx = min(len(absorption) - 1, end_idx + 2)
                
                # Создаем интерполятор на основе нормальных точек вокруг сегмента
                x_normal = np.concatenate([
                    wavelengths[left_idx:start_idx], 
                    wavelengths[end_idx:right_idx]
                ])
                y_normal = np.concatenate([
                    absorption[left_idx:start_idx], 
                    absorption[end_idx:right_idx]
                ])
                
                if len(x_normal) >= 2:  # Минимум 2 точки для интерполяции
                    try:
                        if interpolation_method == 'cubic' and len(x_normal) >= 4:
                            interpolator = CubicSpline(x_normal, y_normal)
                        elif interpolation_method == 'linear':
                            interpolator = interp1d(x_normal, y_normal, kind='linear', 
                                                  fill_value='extrapolate') # type: ignore
                        else:
                            interpolator = interp1d(x_normal, y_normal, kind='quadratic', 
                                                  fill_value='extrapolate') # type: ignore
                        
                        # Интерполируем проблемный сегмент
                        segment_wavelengths = wavelengths[start_idx:end_idx+1]
                        interpolated_values = interpolator(segment_wavelengths)
                        smoothed[start_idx:end_idx+1] = interpolated_values
                        
                    except (ValueError, TypeError):
                        # Если интерполяция не удалась, используем скользящее среднее
                        segment_smooth = uniform_filter1d(
                            absorption[start_idx:end_idx+1], 
                            size=min(5, len(segment))
                        )
                        smoothed[start_idx:end_idx+1] = segment_smooth
    
    # 6. Добавляем результаты в DataFrame
    result_df[result_df.columns[0]] = smoothed
    result_df['is_anomaly'] = anomaly_mask
    
    return result_df

def smooth_uv_spectrum_adaptive_interpolation(df, sensitivity=1.0, min_segment_length=10):
    """
    Адаптивная интерполяция с автоматическим определением параметров
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с колонками 'nm' и 'absorption'
    sensitivity : float
        Чувствительность обнаружения аномалий (0.5-2.0)
    min_segment_length : int
        Минимальная длина сегмента для интерполяции
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame с сглаженным спектром
    """
    
    result_df = df.copy()
    wavelengths = df.index.values
    absorption = df[df.columns[0]].values
    
    # Адаптивное определение параметров
    noise_level = np.std(np.diff(absorption))
    threshold = sensitivity * noise_level * 3
    
    # Находим резкие изменения
    differences = np.abs(np.diff(absorption))
    large_jumps = np.where(differences > threshold)[0]
    
    # Создаем маску для интерполяции
    interpolate_mask = np.zeros(len(absorption), dtype=bool)
    
    for jump_idx in large_jumps:
        start = max(0, jump_idx - 2)
        end = min(len(absorption), jump_idx + 3)
        interpolate_mask[start:end] = True
    
    # Находим сегменты для интерполяции
    segments = []
    current_segment = []
    
    for i in range(len(interpolate_mask)):
        if interpolate_mask[i]:
            current_segment.append(i)
        elif current_segment:
            if len(current_segment) >= min_segment_length:
                segments.append(current_segment)
            current_segment = []
    
    if current_segment and len(current_segment) >= min_segment_length:
        segments.append(current_segment)
    
    # Применяем интерполяцию к каждому сегменту
    smoothed = absorption.copy()
    
    for segment in segments:
        start_idx = max(0, segment[0] - 1)
        end_idx = min(len(absorption) - 1, segment[-1] + 1)
        
        # Точки для интерполяции (до и после сегмента)
        left_idx = max(0, start_idx - 5)
        right_idx = min(len(absorption) - 1, end_idx + 5)
        
        x_reference = np.concatenate([wavelengths[left_idx:start_idx], 
                                    wavelengths[end_idx:right_idx]])
        y_reference = np.concatenate([absorption[left_idx:start_idx], 
                                    absorption[end_idx:right_idx]])
        
        if len(x_reference) >= 4:
            try:
                # Используем кубическую интерполяцию
                interpolator = CubicSpline(x_reference, y_reference)
                segment_wavelengths = wavelengths[start_idx:end_idx+1]
                interpolated = interpolator(segment_wavelengths)
                smoothed[start_idx:end_idx+1] = interpolated
            except:
                # Резервный метод - скользящее среднее
                window = min(7, len(segment))
                smoothed_segment = uniform_filter1d(absorption[start_idx:end_idx+1], size=window)
                smoothed[start_idx:end_idx+1] = smoothed_segment
    
    result_df['is_anomaly'] = interpolate_mask
    result_df[result_df.columns[0]] = smoothed
    
    return result_df

def smooth_uv_spectrum_peaks(df, window_size=5, threshold_factor=2.0, min_peak_height=0.0001):
    """
    Находит и сглаживает участки с резкими перепадами интенсивности в УФ спектре
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с колонками 'nm' (длина волны) и 'absorption' (интенсивность)
    window_size : int
        Размер окна для скользящего среднего (по умолчанию 5)
    threshold_factor : float
        Коэффициент для определения порога перепадов (по умолчанию 2.0)
    min_peak_height : float
        Минимальная высота пика для обнаружения (по умолчанию 0.01)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame с исходными данными и дополнительной колонкой 'absorption_smooth'
        содержащей сглаженный спектр
    """
    
    # Создаем копию DataFrame чтобы не изменять исходные данные
    result_df = df.copy()
    intensity = df[df.columns[0]].values
    
    # 1. Вычисляем первую производную (градиент) для обнаружения резких изменений
    gradient = np.gradient(intensity)
    
    # 2. Находим точки с резкими изменениями интенсивности
    threshold = threshold_factor * np.std(gradient)
    
    # Находим индексы где производная превышает порог (резкие изменения)
    sharp_changes = np.where(np.abs(gradient) > threshold)[0]
    
    # 3. Обнаруживаем пики в исходном сигнале для идентификации проблемных зон
    peaks, _ = signal.find_peaks(intensity, height=min_peak_height, distance=10)
    valleys, _ = signal.find_peaks(-intensity, height=-max(intensity), distance=10)
    
    # Объединяем все проблемные точки
    problematic_indices = np.unique(np.concatenate([sharp_changes, peaks, valleys]))
    
    # 4. Создаем маску для участков с перепадами
    mask = np.zeros(len(intensity), dtype=bool)
    if len(problematic_indices) > 0:
        # Расширяем область вокруг проблемных точек
        for idx in problematic_indices:
            start = max(0, idx - window_size)
            end = min(len(intensity), idx + window_size + 1)
            mask[start:end] = True
    
    # 5. Применяем сглаживание только к проблемным участкам
    smoothed = intensity.copy()
    
    if np.any(mask):
        # Для проблемных участков применяем сглаживание
        smoothed_mask = uniform_filter1d(intensity, size=window_size)
        smoothed[mask] = smoothed_mask[mask]
        
        # Дополнительное сглаживание для очень резких перепадов
        extreme_changes = np.where(np.abs(gradient) > 2 * threshold)[0]
        if len(extreme_changes) > 0:
            for idx in extreme_changes:
                start = max(0, idx - window_size * 2)
                end = min(len(intensity), idx + window_size * 2 + 1)
                # Используем более сильное сглаживание для экстремальных перепадов
                extreme_smooth = uniform_filter1d(intensity[start:end], size=window_size * 2)
                smoothed[start:end] = extreme_smooth
    
    # 6. Добавляем сглаженные данные в DataFrame
    result_df[result_df.columns[0]] = smoothed
    
    return result_df

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from Data_processing.config_paths import INIT_DATA_2024, INIT_DATA_2025, SPECTRA_PATH

date_list = ["2024","2025"]
date_folder_list = [INIT_DATA_2024,INIT_DATA_2025]

for i, folder in enumerate(date_folder_list):
    
    uv_path = Path.joinpath(folder, "UV","Individual_spectra")
    
    uv_abs_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="absorption")
    
    if i == 1:
        uv_refl_spectra = ut.load_spectra_data(uv_path, uv.read_csv_uv, spectra_type="reflection")
        
    for uv_spectra in uv_abs_spectra:
        
        uv_spectra = smooth_uv_spectrum_peaks(uv_spectra, window_size=9, threshold_factor=1.5)
        uv_spectra = uv_spectra.iloc[int(len(uv_spectra) * 8 / 100):]
        
        uv.plot_uv(uv_spectra)
        
        plt.tight_layout()
        plt.savefig(
            Path.joinpath(SPECTRA_PATH,f"{date_list[i]}","UV","Absorption","Individual_spectra",f"{uv_spectra.attrs['name']}")
                    )
        plt.close()