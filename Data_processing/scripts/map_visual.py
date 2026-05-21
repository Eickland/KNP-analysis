import pandas as pd
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import HumSpectra.utilits as ut
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
from shapely.geometry import Point
import matplotlib.patches as mpatches

def create_static_map(df, lat_col='latitude', lon_col='longitude',
                      buffer_km=0.5, title='Точки отбора проб',
                      figsize=(10, 8), marker_size=50, dpi=150):
    """
    Создает статическую карту с точками отбора проб на спутниковом снимке.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с данными проб
    lat_col, lon_col : str
        Названия столбцов с координатами
    buffer_km : float
        Размер буфера вокруг точек в километрах (для определения области карты)
    title : str
        Заголовок карты
    figsize : tuple
        Размер фигуры (ширина, высота) в дюймах
    marker_size : int
        Размер маркеров точек
    dpi : int
        Разрешение сохраняемого изображения
    
    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    
    # Проверяем наличие необходимых столбцов
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Столбцы {lat_col} и/или {lon_col} не найдены в DataFrame")
    
    # Удаляем строки с пропущенными координатами
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    
    if len(df_clean) == 0:
        raise ValueError("Нет данных с валидными координатами")
    
    # Создаем геометрию точек
    geometry = [Point(xy) for xy in zip(df_clean[lon_col], df_clean[lat_col])]
    
    # Создаем GeoDataFrame в WGS84 (EPSG:4326)
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
    
    # Конвертируем в проекцию для расчета расстояний (метры)
    # Используем UTM зону или просто меркатор
    gdf_projected = gdf.to_crs("EPSG:3857")  # Web Mercator
    
    # Определяем границы карты
    bounds = gdf_projected.total_bounds
    xmin, ymin, xmax, ymax = bounds
    
    # Добавляем буфер в метрах
    buffer_m = buffer_km * 1000
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m
    
    # Создаем фигуру и оси
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Конвертируем точки обратно в WGS84 для отображения на спутниковом снимке
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    
    # Отображаем точки
    gdf_wgs84.plot(
        ax=ax,
        color='red',
        marker='o',
        markersize=marker_size,
        edgecolor='white',
        linewidth=1,
        zorder=5,
        label='Точки отбора'
    )
    
    # Добавляем номера точек (если есть столбец sample_id)
    if 'sample_id' in gdf_wgs84.columns:
        for idx, row in gdf_wgs84.iterrows():
            ax.annotate(
                row['sample_id'],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                color='white',
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7)
            )
    
    # Добавляем подложку OpenStreetMap (спутник)
    try:
        ctx.add_basemap(
            ax,
            crs=gdf_wgs84.crs,
            source=ctx.providers.OpenStreetMap.Mapnik,  # Обычная карта
            # source=ctx.providers.Esri.WorldImagery,  # Спутник (закомментировано)
            # source=ctx.providers.Google.Satellite,   # Google спутник (требуется ключ)
            alpha=0.8
        )
    except Exception as e:
        print(f"Не удалось загрузить подложку: {e}")
        print("Попробую использовать простую карту...")
        ctx.add_basemap(
            ax,
            crs=gdf_wgs84.crs,
            source=ctx.providers.CartoDB.Positron,  # Светлая карта как запасной вариант
            alpha=0.8
        )
    
    # Настраиваем отображение
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Долгота', fontsize=10)
    ax.set_ylabel('Широта', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    # Добавляем масштабную линейку
    scale_bar = mpatches.Patch(
        color='none', 
        label=f'Масштаб: ~{buffer_km * 2:.1f} км',
        visible=False
    )
    ax.legend(handles=[scale_bar], loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    return fig, ax


def create_satellite_map_simple(df, lat_col='latitude', lon_col='longitude',
                                 buffer_km=0.5, title='Точки отбора проб',
                                 output_file='sampling_map.png',
                                 use_satellite=True):
    """
    Упрощенная функция для быстрого создания карты.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame с данными
    lat_col, lon_col : str
        Названия столбцов координат
    buffer_km : float
        Буфер вокруг точек в км
    title : str
        Заголовок
    output_file : str
        Имя файла для сохранения
    use_satellite : bool
        True - спутниковый снимок, False - обычная карта
    """
    
    # Проверка данных
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    
    if len(df_clean) == 0:
        print("Нет валидных координат!")
        return None, None
    
    # Создаем GeoDataFrame
    geometry = [Point(xy) for xy in zip(df_clean[lon_col], df_clean[lat_col])]
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")
    
    # Определяем границы
    bounds = gdf.total_bounds
    xmin, ymin, xmax, ymax = bounds
    
    # Добавляем буфер в градусах (приблизительно)
    buffer_deg = buffer_km / 111  # 1 градус ≈ 111 км
    xmin -= buffer_deg
    xmax += buffer_deg
    ymin -= buffer_deg
    ymax += buffer_deg
    
    # Создаем карту
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Рисуем точки
    gdf.plot(
        ax=ax,
        color='red',
        marker='o',
        markersize=12,
        edgecolor='white',
        linewidth=2,
        zorder=5,
        label='Точки отбора'
    )
    """
    # Добавляем подписи
    for idx, row in gdf.iterrows():
        ax.annotate(
            f"{idx+1}",
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=6,
            color='white',
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8)
        )
    """
    # Выбираем источник подложки
    if use_satellite:
        source = ctx.providers.Esri.WorldImagery  # Спутник
    else:
        source = ctx.providers.OpenStreetMap.Mapnik  # Обычная карта
    
    # Добавляем подложку
    try:
        ctx.add_basemap(ax, crs=gdf.crs, source=source, alpha=0.9)
    except Exception as e:
        print(f"Ошибка загрузки подложки: {e}")
        print("Использую запасной вариант...")
        ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron)
    
    # Настраиваем границы
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Оформление
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Долгота', fontsize=11)
    ax.set_ylabel('Широта', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(loc='upper right', fontsize=11)
    
    # Добавляем информацию о количестве проб
    textstr = f'Всего проб: {len(gdf)}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Сохраняем
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Карта сохранена в {output_file}")
    
    # Показываем
    plt.show()
    
    return fig, ax

# Пример использования
if __name__ == "__main__":
    # Создаем пример данных
    df = pd.read_excel(r"D:\lab\KNP-analysis\Initial_data\Additional_data\GeoData.xlsx")
    df['Class'] = df.Sample.apply(lambda x: ut.extract_subclass_from_name(x))
    df = df[df['Sample'].str.contains('2025')]

    
    #df = df[df['Class'] == 'Baikal']
    df = df[df['latitude'] < 51.6]
    df = df[df['latitude'] != 51.496572]
    df.reset_index(inplace=True)
    # Создаем карту
    map_obj = create_satellite_map_simple(
        df=df,
        lat_col='latitude',
        lon_col='longitude',
        buffer_km=0.3,  # 300 метров вокруг точек
        title='Точки отбора проб',
        output_file='sampling_map_satellite.png',
        use_satellite=True  # True - спутник, False - обычная карта
    )
    
    # Сохраняем карту
    plt.tight_layout()
    plt.savefig(r"D:\lab\KNP-analysis\Initial_data\Additional_data\Байкал.png", dpi=300, bbox_inches='tight')
    # Отображаем карту (если работаем в Jupyter notebook)
    # display(map_obj)