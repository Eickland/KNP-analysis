import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from contextlib import contextmanager

import HumSpectra.utilits as ut
import HumSpectra.mass_descriptors as md

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.titlesize'] = 16
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams["legend.fontsize"] = 17


@contextmanager
def timer(name=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"{name}: {end - start:.4f} сек")

wdir_spectra = r"D:\lab\MassSpectraNew\processed_results\mzlist"
wdir_statistic = r"D:\lab\MassSpectraNew\processed_results\statistic"
wdir = r"D:\lab\MassSpectraNew\processed_results"

bad_spectra_data = pd.read_csv(r"D:\lab\MassSpectraNew\processed_results\bad_spectra.csv")
bad_spectra_list = bad_spectra_data['name'].to_list()

spectra_list = []
spectra_stat_list = []
spectra_class_list = []
spectra_list_name = []

with timer("Чтение спектров"):
    for path in Path.rglob(Path(wdir_spectra),'*.csv'):
        
        spectra = pd.read_csv(path)
        spectra_name = ut.extract_name_from_path(str(path))
        spectra_name = spectra_name.split(sep='__')[0]
        
        if spectra_name in bad_spectra_list or spectra_name == 'ADOM-SL9-6':
            continue
        
        spectra_class = ut.extract_class_from_name(spectra_name)
        spectra.attrs['name'] = spectra_name
        spectra.attrs['class'] = spectra_class
        
        spectra_list.append(spectra)
        spectra_class_list.append(spectra_class)
        spectra_list_name.append(spectra_name)

metric_data = md.calculate_metrics_for_spectra_list(spectra_list=spectra_list, metrics=None)
print(metric_data.columns)
metric_data.drop(columns=['S','C_13','mass'],inplace=True)
metric_data['subclass'] = metric_data.sample_name.apply(lambda x: ut.extract_subclass_from_name(x))
metric_data = metric_data[metric_data['subclass'] != 'Storage 1']
if 'DBE_AI' not in metric_data.columns:
    metric_data['DBE_AI'] = metric_data['AI']  # или df['DBE_AI'] = df['AI_mod'] если есть

# Группировка по образцам для расчета статистик
grouped = metric_data.groupby(['sample_name', 'subclass'])

# Функция аггрегации метрик для одного образца
def agg_metrics(x):
    return pd.Series({
        'HC': x['H/C'].mean(),
        'OC': x['O/C'].mean(),
        'NOSC': x['NOSC'].mean(),
        'DBE': x['DBE'].mean(),
        'AI': x['AI'].mean(),
        'DBE_AI': x['DBE_AI'].mean(),
        'CRAM': x['CRAM'].mean()  # т.к. CRAM bool -> среднее = доля True
    })

sample_stats = grouped.apply(agg_metrics).reset_index()

# Также можно получить сводную статистику по классам
class_stats = sample_stats.groupby('subclass').agg({
    'HC': ['mean', 'std', 'median'],
    'OC': ['mean', 'std', 'median'],
    'NOSC': ['mean', 'std', 'median'],
    'DBE': ['mean', 'std', 'median'],
    'AI': ['mean', 'std', 'median'],
    'DBE_AI': ['mean', 'std', 'median'],
    'CRAM': ['mean', 'std', 'median']
}).round(4)

metrics = ['HC', 'OC', 'NOSC', 'DBE', 'AI', 'DBE_AI', 'CRAM']
subclasses = class_stats.index.tolist()

# Настройка сетки графиков (2x4 или 3x3)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    if idx < len(axes):
        means = class_stats[metric]['mean']
        stds = class_stats[metric]['std']
        
        # Построение столбцов
        x_pos = np.arange(len(subclasses))
        axes[idx].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                      color='steelblue', ecolor='black')
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(subclasses, rotation=45, ha='right')
        axes[idx].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Mean ± Std')
        axes[idx].grid(axis='y', alpha=0.3)

# Удалить лишний subplot если есть
if len(metrics) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()

from math import pi
mean_matrix = class_stats.xs('mean', axis=1, level=1)
# Нормализуем данные для радарного графика
normalized = (mean_matrix - mean_matrix.min()) / (mean_matrix.max() - mean_matrix.min())

# Углы для радара
angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]  # Замыкаем круг

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

for subclass in subclasses:
    values = normalized.loc[subclass].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=subclass)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title('Normalized metric profiles by subclass', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

def plot_square_intensities_by_subclass(
    square_data: pd.DataFrame,
    square_indices: Optional[List[int]] = None,
    group_by: str = 'class',  # 'class' or 'sample_name'
    plot_type: str = 'heatmap',  # 'heatmap', 'bar', 'box', 'violin', 'stacked'
    normalize_by_subclass: bool = False,  # Normalize across subclasses
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_values: bool = True,
    cmap: str = 'viridis',
    **kwargs
):
    """
    Visualize square intensities across subclasses.
    
    Parameters
    ----------
    square_data : pd.DataFrame
        DataFrame from extract_square_intensities
    square_indices : List[int], optional
        Specific squares to plot. If None, uses all squares
    group_by : str, default='class'
        Column name for grouping ('class' or 'subclass')
    plot_type : str, default='heatmap'
        'heatmap', 'bar', 'box', 'violin', 'stacked'
    normalize_by_subclass : bool, default=False
        Normalize intensities across subclasses (for heatmap)
    figsize : Tuple[int, int], default=(12, 8)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    show_values : bool, default=True
        Show values on heatmap
    cmap : str, default='viridis'
        Colormap for heatmap
    """
    
    # Identify square columns (numeric columns not 'name', 'sample_name', 'class')
    exclude_cols = ['name', 'sample_name', 'class']
    square_cols = [col for col in square_data.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(square_data[col])]
    
    # Filter selected squares
    if square_indices is not None:
        square_cols = [col for col in square_cols 
                       if int(col) in square_indices]
    
    # Group by the specified column and calculate mean and std
    grouped_stats = square_data.groupby(group_by)[square_cols].agg(['mean', 'std']).round(4)
    
    # For heatmap: calculate mean intensities
    heatmap_data = square_data.groupby(group_by)[square_cols].mean()
    
    if normalize_by_subclass:
        # Normalize across subclasses (each row sums to 1)
        heatmap_data = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    # Create plot based on type
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'heatmap':
        # Heatmap of mean intensities
        im = sns.heatmap(heatmap_data, annot=show_values, fmt='.3f', 
                        cmap=cmap, ax=ax, cbar_kws={'label': 'Mean Intensity'})
        ax.set_title(f'Square Intensities by {group_by}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Square Index')
        ax.set_ylabel(group_by.capitalize())
        
    elif plot_type == 'bar':
        # Grouped bar plot with error bars
        x = np.arange(len(square_cols))
        width = 0.8 / len(grouped_stats.index)
        colors = plt.colormaps['tab10'](np.linspace(0, 1, len(grouped_stats.index)))
        
        for i, (subclass, row) in enumerate(grouped_stats.iterrows()):
            means = [row[col]['mean'] for col in square_cols]
            stds = [row[col]['std'] for col in square_cols]
            offset = (i - len(grouped_stats.index)/2) * width
            ax.bar(x + offset, means, width, label=subclass, 
                  yerr=stds, capsize=3, color=colors[i], alpha=0.7)
        
        ax.set_xlabel('Square Index')
        ax.set_ylabel('Mean Intensity')
        ax.set_title(f'Square Intensities by {group_by}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(square_cols)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
    elif plot_type == 'box':
        # Box plot for selected squares
        melted_data = square_data.melt(id_vars=[group_by], 
                                       value_vars=square_cols,
                                       var_name='square', 
                                       value_name='intensity')
        sns.boxplot(data=melted_data, x='square', y='intensity', 
                   hue=group_by, ax=ax)
        ax.set_title(f'Square Intensity Distribution by {group_by}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
    elif plot_type == 'violin':
        # Violin plot for distribution
        melted_data = square_data.melt(id_vars=[group_by], 
                                       value_vars=square_cols,
                                       var_name='square', 
                                       value_name='intensity')
        sns.violinplot(data=melted_data, x='square', y='intensity', 
                      hue=group_by, split=True, ax=ax)
        ax.set_title(f'Square Intensity Distribution by {group_by}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    elif plot_type == 'stacked':
        # Stacked bar plot (relative abundance)
        stacked_data = heatmap_data.copy()
        if not normalize_by_subclass:
            stacked_data = stacked_data.div(stacked_data.sum(axis=1), axis=0)
        
        stacked_data.plot(kind='bar', stacked=True, ax=ax, colormap=cmap)
        ax.set_title(f'Relative Square Abundance by {group_by}', fontsize=14, fontweight='bold')
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel('Relative Intensity')
        ax.legend(title='Square', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_comparison_of_selected_squares(
    square_data: pd.DataFrame,
    squares_dict: Dict[str, List[int]],  # e.g., {'Region 1': [1,2,3], 'Region 2': [4,5,6]}
    group_by: str = 'class',
    plot_type: str = 'bar',  # 'bar', 'box'
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare aggregated groups of squares by subclass.
    
    Parameters
    ----------
    square_data : pd.DataFrame
        DataFrame from extract_square_intensities
    squares_dict : Dict[str, List[int]]
        Dictionary mapping region names to lists of square indices
    group_by : str, default='class'
        Column name for grouping
    plot_type : str, default='bar'
        'bar' or 'box'
    figsize : Tuple[int, int], default=(10, 6)
        Figure size
    save_path : Optional[str], default=None
        Path to save figure
    """
    
    # Create aggregated columns for each region
    data_copy = square_data.copy()
    
    for region_name, squares in squares_dict.items():
        square_cols = [str(s) for s in squares if str(s) in square_data.columns]
        if square_cols:
            data_copy[region_name] = square_data[square_cols].sum(axis=1)
    
    # Select only the aggregated columns
    agg_cols = list(squares_dict.keys())
    
    # Melt for plotting
    melted_data = data_copy.melt(id_vars=[group_by], 
                                 value_vars=agg_cols,
                                 var_name='region', 
                                 value_name='intensity')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'bar':
        sns.barplot(data=melted_data, x='region', y='intensity', 
                   hue=group_by, ax=ax, capsize=0.1)
        ax.set_title('Regional Square Intensities by Subclass', fontsize=14, fontweight='bold')
        
    elif plot_type == 'box':
        sns.boxplot(data=melted_data, x='region', y='intensity', 
                   hue=group_by, ax=ax)
        ax.set_title('Regional Square Intensity Distribution by Subclass', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Total Intensity')
    ax.set_xlabel('Region')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig, ax

square_data = md.extract_square_intensities(spectra_list,hc_range=(0.5,2.0),oc_range=(0.2,0.8))
# Add subclass information if not present
if 'subclass' not in square_data.columns:
    # Assuming you have a function to extract subclass from sample_name
    square_data['subclass'] = square_data['sample_name'].apply(
        lambda x: ut.extract_subclass_from_name(x)
    )

# Example 1: Heatmap of all squares by subclass
plot_square_intensities_by_subclass(
    square_data,
    plot_type='heatmap',
    group_by='subclass',
    figsize=(14, 6),
    show_values=True,
    cmap='YlOrRd'
)

# Example 2: Bar plot for specific squares with error bars
plot_square_intensities_by_subclass(
    square_data,
    square_indices=[5, 6, 11, 12, 16],  # First 5 squares
    plot_type='bar',
    group_by='subclass',
    figsize=(12, 6)
)

# Example 3: Box plots for distribution
plot_square_intensities_by_subclass(
    square_data,
    square_indices=[5, 6, 11, 12, 16],
    plot_type='box',
    group_by='subclass',
    figsize=(10, 6)
)

# Example 4: Stacked bar for relative abundances
plot_square_intensities_by_subclass(
    square_data,
    plot_type='stacked',
    group_by='subclass',
    figsize=(12, 6),
    normalize_by_subclass=True
)

# Example 5: Compare aggregated regions
regions = {
    'Low O/C': [1, 2, 3, 4],
    'Mid O/C': [5, 6, 7, 8],
    'High O/C': [9, 10, 11, 12]
}

plot_comparison_of_selected_squares(
    square_data,
    squares_dict=regions,
    group_by='subclass',
    plot_type='bar',
    figsize=(10, 6)
)
