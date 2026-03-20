import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import List, Tuple
import sys
import pandas as pd
from pathlib import Path

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
from HumSpectra.sample import SampleCollection

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
from Data_processing.Utilits.config_paths import SPECTRA_PATH, CACHE_DATA_PATH

plt.rcParams.update({
    'font.size': 10,                    # Базовый размер шрифта
    'axes.titlesize': 10,               # Заголовок осей
    'axes.labelsize': 10,               # Метки осей
    'xtick.labelsize': 10,              # Метки на оси X
    'ytick.labelsize': 10,              # Метки на оси Y
    'legend.fontsize': 10,              # Легенда
    'figure.titlesize': 10              # Заголовок фигуры
})

class PDFCatalog:
    def __init__(self, output_path: str|Path = "spectra_catalog.pdf"):
        """
        Инициализация каталога спектров
        
        Args:
            output_filename: имя выходного PDF файла
        """
        self.output_path = output_path
        self.pdf_pages = None
        
    def create_plot_grid(self, fig, spectra_data, start_idx: int):
        """
        Создание сетки графиков на одной странице
        
        Args:
            fig: объект фигуры matplotlib
            spectra_data: список данных спектров
            start_idx: начальный индекс спектра для текущей страницы
        """
        # Создаем 3 строки по 3 графика в каждой (всего 9 графиков на страницу)
        # Но у нас 3 спектра на страницу, каждый спектр занимает одну строку из 3 графиков
        
        spectra_on_page = min(3, len(spectra_data) - start_idx)
        
        for i in range(spectra_on_page):
            spectrum_idx = start_idx + i
            spectrum_data = spectra_data[spectrum_idx]
            
            # Создаем 3 подграфика для каждого спектра в одной строке
            for j in range(3):
                ax = fig.add_subplot(3, 3, i * 3 + j + 1)
                
                # Здесь будет вызов ваших функций для рисования
                self._plot_spectrum_part(ax, spectrum_data, j)
                
        # Общие настройки для страницы
        #fig.suptitle(f'Каталог спектров (страница {start_idx//3 + 1})', fontsize=14)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.tight_layout()
    
    def _plot_spectrum_part(self, ax, spectrum_data, plot_type: int):
        """
        Метод для рисования отдельных частей спектра.
        Этот метод нужно переопределить в вашем коде.
        
        Args:
            ax: ось для рисования
            spectrum_data: данные спектра
            plot_type: тип графика (0, 1, 2 для трех разных представлений)
        """
        
        if plot_type == 0:
            
            fl.plot_heat_map(spectrum_data[0],ax=ax)
            ax.set_title('3D спектр флуорсценции')
            
        elif plot_type == 1:
            
            uv.plot_uv(spectrum_data[1],ax=ax,norm_by_TOC=True)
            ax.set_title(f'{spectrum_data[1].attrs['name']}\nСпектр поглощения')
            ax.set_xlabel('Длина волны, нм')
            
        elif plot_type == 2:
            
            uv.plot_uv(spectrum_data[2],ax=ax)
            ax.set_title('Спектр отражения')
            ax.set_ylabel('R, % отражения')
            ax.set_xlabel('Длина волны, нм')
            ax.set_ylim((0,100))
    
    def generate_catalog(self, spectra_data: List):
        """
        Генерация PDF каталога
        
        Args:
            spectra_data: список данных спектров
        """
        with PdfPages(self.output_path) as pdf:
            total_spectra = len(spectra_data)
            
            # Обрабатываем по 3 спектра на страницу
            for page_start in range(0, total_spectra, 3):
                # Создаем новую страницу A4
                fig = plt.figure(figsize=(11.69, 8.27))  # A4 в дюймах
                
                # Заполняем страницу графиками
                self.create_plot_grid(fig, spectra_data, page_start)
                
                # Сохраняем страницу в PDF
                pdf.savefig(fig)
                plt.close(fig)
                
                print(f"Обработано {min(page_start + 3, total_spectra)} из {total_spectra} спектров")
                
pdf_creator = PDFCatalog(Path.joinpath(SPECTRA_PATH,'catalog_spectra.pdf'))

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

sample_list = collection.filter_by_tags_all('cutted_fluo','smooth_uv_refl','smooth_uv_abs','org_carbon')
print(f'Коллекция профильтрвана, к-во образцов: {len(sample_list)}')

spectra_list = []

for sample in sample_list:
    
    if sample.uv_vis_absorption_smooth is None:
        raise ValueError('Ошибка спектра поглощения')
    
    sample.uv_vis_absorption_smooth.attrs['TOC'] = sample.org_carbon
    
    spectra_list.append([sample.cutted_fluo_eem,sample.uv_vis_absorption_smooth,sample.uv_vis_reflection_smooth])
    
pdf_creator.generate_catalog(spectra_list)