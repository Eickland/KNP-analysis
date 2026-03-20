import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from typing import List, Tuple
import sys
import pandas as pd
from pathlib import Path

import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
from HumSpectra.sample import SampleCollection, split_by_category

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
from Data_processing.Utilits.config_paths import SPECTRA_PATH, CACHE_DATA_PATH

plt.rcParams.update({
    'font.size': 10,                    # Базовый размер шрифта
})

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

for sample_name, sample in collection.samples.items():
    
    print(sample.sample_subclass)

by_subclass = split_by_category(collection, category='subclass')

sample_list = collection.filter_by_tags_all('cutted_fluo','smooth_uv_refl','smooth_uv_abs','org_carbon')
print(f'Коллекция профильтрвана, к-во образцов: {len(sample_list)}')

by_subclass = split_by_category(sample_list, category='subclass')
print(f"Найдено подклассов: {len(by_subclass)}")
for subclass, samples in by_subclass.items():
    print(f"  {subclass}: {len(samples)} образцов")

subclass_list = list(by_subclass.items())

fig, axes = plt.subplots(4, 3, figsize=(14, 12),dpi=300)

# Удаляем лишние subplots в последней строке
fig.delaxes(axes[3, 2])  # Удаляем правый нижний
fig.delaxes(axes[3, 0])  # Удаляем левый нижний
fig.delaxes(axes[3, 1])  # Удаляем центральный нижний (освобождаем место)

# Заполняем первые 9 графиков
for i in range(9):
    row = i // 3
    col = i % 3
    
    for k in range(3):
        subclass_list[i][1][k].uv_vis_absorption_smooth.attrs['TOC'] = subclass_list[i][1][k].org_carbon #type: ignore
        uv.plot_uv(subclass_list[i][1][k].uv_vis_absorption_smooth, #type: ignore
                  ax=axes[row, col],
                  name=subclass_list[i][1][k].uv_vis_absorption_smooth.attrs['name'], #type: ignore
                  norm_by_TOC=True)
    
    # Добавляем легенду для каждого subplot
    axes[row, col].legend()
    axes[row, col].set_title(f'{subclass_list[i][0]}')
    axes[row, col].set_xlabel('Длина волны, нм')

plt.subplots_adjust(wspace=0.4, hspace=0.5)

# Рассчитываем позицию для 10-го графика
# Используем позицию удаленного axes[3, 0] как базовую
ref_pos = axes[2, 0].get_position()  # Используем позицию из третьей строки для выравнивания

# Первый график в нижнем ряду (10-й)
left_pos = ref_pos.x0 - 0.12
bottom = ref_pos.y0 - ref_pos.height * 2  # Смещаем ниже
width = ref_pos.width
height = ref_pos.height

ax10 = fig.add_axes((left_pos, bottom, width, height))

for k in range(3):
    subclass_list[9][1][k].uv_vis_absorption_smooth.attrs['TOC'] = subclass_list[9][1][k].org_carbon #type: ignore
    uv.plot_uv(subclass_list[9][1][k].uv_vis_absorption_smooth, #type: ignore
              ax=ax10,
              name=subclass_list[9][1][k].uv_vis_absorption_smooth.attrs['name'], #type: ignore
              norm_by_TOC=True)

ax10.legend()
ax10.set_title(f'{subclass_list[9][0]}')
ax10.set_xlabel('Длина волны, нм')

# Второй график в нижнем ряду (11-й)
# Используем позицию axes[2, 1] для выравнивания
ref_pos2 = axes[2, 1].get_position()
left_pos2 = ref_pos2.x0 - 0.03
bottom2 = bottom  # Такая же высота как у ax10
width2 = ref_pos2.width
height2 = height

ax11 = fig.add_axes((left_pos2, bottom2, width2, height2))

for k in range(3):
    subclass_list[10][1][k].uv_vis_absorption_smooth.attrs['TOC'] = subclass_list[10][1][k].org_carbon #type: ignore
    uv.plot_uv(subclass_list[10][1][k].uv_vis_absorption_smooth, #type: ignore
              ax=ax11,
              name=subclass_list[10][1][k].uv_vis_absorption_smooth.attrs['name'], #type: ignore
              norm_by_TOC=True)

ax11.legend()
ax11.set_title(f'{subclass_list[10][0]}')
ax11.set_xlabel('Длина волны, нм')

# Опционально: центрируем два нижних графика
# Вычисляем общую ширину двух графиков + промежуток
total_width = width + width2 + (ref_pos2.x0 - ref_pos.x0 - width)
center_offset = (1 - total_width) / 2

# Смещаем оба графика для центрирования
ax10.set_position((ax10.get_position().x0 + center_offset, 
                   ax10.get_position().y0,
                   ax10.get_position().width,
                   ax10.get_position().height))

ax11.set_position((ax11.get_position().x0 + center_offset,
                   ax11.get_position().y0,
                   ax11.get_position().width,
                   ax11.get_position().height))

plt.tight_layout()
plt.savefig(r"C:\Users\mnbv2\Desktop\lab\KNP-analysis\Server\Received_data\Spectra\2025\UV\Absorption\Спектры по подклассам.png")
plt.show()