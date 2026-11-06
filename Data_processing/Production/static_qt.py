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

data = pd.read_excel(r"D:\lab\KNP-analysis\Received_data\Statistics\2024\Documents\Исходные данные за 2024 год.xlsx")


ph_data = pd.read_csv(r'D:\lab\KNP-analysis\Initial_data\Additional_data\GeoData.csv')

main = data.merge(ph_data, on='Sample')
main.drop(columns=['latitude', 'longitude'], inplace=True)
print(main.columns)
main=data

# ============================================================================
# КВАРТИЛИ ПО КЛАССАМ И ПОДКЛАССАМ
# ============================================================================

OUTPUT_PATH = Path(r"D:\lab\KNP-analysis\Received_data\Statistics\2024\Documents\Квартили за 2024 год.xlsx")
DECIMALS = 3          # знаков после запятой в таблицах
ADD_TOTAL_ROW = True  # добавить строку "Все образцы" (по всему набору без деления на группы)

numeric_cols = list(main.select_dtypes(include=[np.number]).columns)

# Порядок статистик в итоговой таблице
STATS = ['n', 'Min', 'Q1', 'Медиана', 'Q3', 'Max', 'IQR']


def quartile_table(df: pd.DataFrame, group_cols: list, add_total: bool = ADD_TOTAL_ROW) -> pd.DataFrame:
    """
    Квартили по группам.
    Строки - группы (класс или класс+подкласс), колонки - MultiIndex (признак, статистика).
    Квартили считаются линейной интерполяцией (как QUARTILE.INC / КВАРТИЛЬ.ВКЛ в Excel).
    NaN не учитываются, n - число непустых значений признака в группе.
    """
    def _stats(g) -> pd.DataFrame:
        q1 = g.quantile(0.25)
        q3 = g.quantile(0.75)
        res = pd.concat({
            'n': g.count(),
            'Min': g.min(),
            'Q1': q1,
            'Медиана': g.median(),
            'Q3': q3,
            'Max': g.max(),
            'IQR': q3 - q1,
        }, axis=1)                                  # колонки: (статистика, признак)
        res = res.swaplevel(0, 1, axis=1)           # -> (признак, статистика)
        res = res.reindex(columns=pd.MultiIndex.from_product([numeric_cols, STATS]))
        return res

    table = _stats(df.groupby(group_cols)[numeric_cols])

    if add_total:
        total_key = tuple(['Все образцы'] + [''] * (len(group_cols) - 1)) if len(group_cols) > 1 else 'Все образцы'
        total = _stats(df.assign(_all=0).groupby('_all')[numeric_cols])
        total.index = pd.MultiIndex.from_tuples([total_key], names=group_cols) if len(group_cols) > 1 \
            else pd.Index([total_key], name=group_cols[0])
        table = pd.concat([table, total])

    # n - целое, остальное округляем
    for feature in numeric_cols:
        table[(feature, 'n')] = table[(feature, 'n')].astype(int)
    stat_cols = [c for c in table.columns if c[1] != 'n']
    table[stat_cols] = table[stat_cols].round(DECIMALS)
    return table


def compact_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Компактный вид для статьи/отчета: 'Медиана [Q1; Q3]' в одной ячейке, n отдельно.
    """
    out = pd.DataFrame(index=table.index)
    for feature in numeric_cols:
        med = table[(feature, 'Медиана')]
        q1 = table[(feature, 'Q1')]
        q3 = table[(feature, 'Q3')]
        out[feature] = [
            '' if pd.isna(m) else f"{m:.{DECIMALS}g} [{a:.{DECIMALS}g}; {b:.{DECIMALS}g}]"
            for m, a, b in zip(med, q1, q3)
        ]
    # n - число образцов в группе (по строкам таблицы, максимум по признакам)
    out.insert(0, 'n', table.xs('n', axis=1, level=1).max(axis=1).astype(int))
    return out


def long_table(table: pd.DataFrame) -> pd.DataFrame:
    """Длинный формат: группа | признак | n | Min | Q1 | Медиана | Q3 | Max | IQR (удобно фильтровать)."""
    long = table.stack(level=0, future_stack=True) if 'future_stack' in pd.DataFrame.stack.__code__.co_varnames \
        else table.stack(level=0)
    long.index = long.index.set_names(list(table.index.names) + ['Признак'])
    return long[STATS].reset_index()


q_class = quartile_table(main, ['Class'])
q_subclass = quartile_table(main, ['Class', 'Subclass'])

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)
print("\nКВАРТИЛИ ПО КЛАССАМ (Медиана [Q1; Q3]):")
print(compact_table(q_class).to_string())
print("\nКВАРТИЛИ ПО ПОДКЛАССАМ (Медиана [Q1; Q3]):")
print(compact_table(q_subclass).to_string())

with pd.ExcelWriter(OUTPUT_PATH) as writer:
    compact_table(q_class).to_excel(writer, sheet_name='Классы (кратко)')
    compact_table(q_subclass).to_excel(writer, sheet_name='Подклассы (кратко)')
    q_class.to_excel(writer, sheet_name='Классы (полностью)')
    q_subclass.to_excel(writer, sheet_name='Подклассы (полностью)')
    long_table(q_class).to_excel(writer, sheet_name='Классы (длинный)', index=False)
    long_table(q_subclass).to_excel(writer, sheet_name='Подклассы (длинный)', index=False)

print(f"\n✅ Квартили сохранены: {OUTPUT_PATH}")