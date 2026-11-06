"""
Isolation Forest: модель "нормы" обучается только на классе ADOM,
затем ищутся аномалии (несоответствие ADOM) в классе Baikal (тестовый)
и во внешних точках Tech и Osinovka. Остальные классы не используются.

Два набора признаков:
  * all        - все 11 дескрипторов как есть;
  * intensive  - только признаки, не зависящие от концентрации ОВ
                 (отношения/индексы + доли PARAFAC-компонент вместо Fmax).
Зачем второй набор: у Baikal C_org, epsilon и Fmax компонент в 5-10 раз ниже,
чем у ADOM, поэтому на полном наборе "аномальность" почти целиком определяется
концентрацией, а не составом ОВ.

Порог аномальности: 5-й перцентиль out-of-fold скоров ADOM
(каждый образец ADOM оценивается моделью, не видевшей его при обучении),
т.е. ожидаемая доля ложных тревог на ADOM ~5%.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(r"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Received_data\Statistics\2024\Documents\Исходные данные за 2024 год.xlsx")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else DATA_PATH.parent

TRAIN_CLASS = 'ADOM'
TEST_CLASS = 'Baikal'
EXTERNAL_CLASSES = ['Tech', 'Osinovka']

N_ESTIMATORS = 1000
RANDOM_STATE = 42
FALSE_ALARM_RATE = 0.05     # доля ADOM, которую допускаем считать аномалией
N_FOLDS = 5

ALL_FEATURES = ['E2E3', 'epsilon', 'C_org', 'Suva', 'Asm_350', 'Asm_280',
                'FIX', 'HIX', 'Component_1', 'Component_2', 'Component_3']
INTENSIVE_BASE = ['E2E3', 'Suva', 'Asm_350', 'Asm_280', 'FIX', 'HIX']
COMPONENTS = ['Component_1', 'Component_2', 'Component_3']


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
data = pd.read_excel(DATA_PATH)
data = data[data['Class'].isin([TRAIN_CLASS, TEST_CLASS] + EXTERNAL_CLASSES)].copy()

# Доли компонент (не зависят от концентрации)
comp_sum = data[COMPONENTS].sum(axis=1)
for c in COMPONENTS:
    data[f'{c}_frac'] = data[c] / comp_sum
INTENSIVE_FEATURES = INTENSIVE_BASE + [f'{c}_frac' for c in COMPONENTS]

FEATURE_SETS = {'all': ALL_FEATURES, 'intensive': INTENSIVE_FEATURES}

train_df = data[data['Class'] == TRAIN_CLASS]
eval_df = data[data['Class'] != TRAIN_CLASS]
print(f"Обучение: {TRAIN_CLASS} - {len(train_df)} образцов "
      f"({train_df['Subclass'].nunique()} подклассов)")
print(eval_df['Class'].value_counts().to_string(), '\n')


def make_model():
    return IsolationForest(n_estimators=N_ESTIMATORS, contamination='auto',
                           random_state=RANDOM_STATE, n_jobs=-1)


def adom_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Доля ADOM (out-of-fold) с таким же или более низким скором = эмпирическое p-значение."""
    return np.array([(reference <= v).mean() * 100 for v in values])


def features_out_of_range(row: pd.Series, ref: pd.DataFrame, features) -> str:
    """Признаки, значение которых вне диапазона P2.5-P97.5 ADOM (с направлением)."""
    lo, hi = ref[features].quantile(0.025), ref[features].quantile(0.975)
    out = []
    for f in features:
        if row[f] < lo[f]:
            out.append(f'{f}↓')
        elif row[f] > hi[f]:
            out.append(f'{f}↑')
    return ', '.join(out)


results = []
summary = []
oof_store = {}

for set_name, features in FEATURE_SETS.items():
    X_train = train_df[features].values

    # 1. Out-of-fold скоры ADOM (честная оценка "нормы")
    oof = np.empty(len(train_df))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(X_train, train_df['Subclass']):
        m = make_model().fit(X_train[tr])
        oof[te] = m.score_samples(X_train[te])
    threshold = np.quantile(oof, FALSE_ALARM_RATE)
    oof_store[set_name] = (oof, threshold)

    # 2. Финальная модель на всём ADOM
    model = make_model().fit(X_train)
    eval_scores = model.score_samples(eval_df[features].values)

    # ADOM (out-of-fold)
    for (_, row), s in zip(train_df.iterrows(), oof):
        results.append({
            'Набор признаков': set_name, 'Class': row['Class'], 'Subclass': row['Subclass'],
            'Sample': row['Sample'], 'Роль': 'обучение (out-of-fold)', 'Score': s,
            'Перцентиль среди ADOM, %': adom_percentile([s], oof)[0],
            'Аномалия': s < threshold,
            'Признаки вне диапазона ADOM': features_out_of_range(row, train_df, features),
        })
    # Baikal, Tech, Osinovka
    for (_, row), s in zip(eval_df.iterrows(), eval_scores):
        role = 'тест' if row['Class'] == TEST_CLASS else 'внешняя точка'
        results.append({
            'Набор признаков': set_name, 'Class': row['Class'], 'Subclass': row['Subclass'],
            'Sample': row['Sample'], 'Роль': role, 'Score': s,
            'Перцентиль среди ADOM, %': adom_percentile([s], oof)[0],
            'Аномалия': s < threshold,
            'Признаки вне диапазона ADOM': features_out_of_range(row, train_df, features),
        })

    # 3. Сводка по классам
    for cls in [TRAIN_CLASS, TEST_CLASS] + EXTERNAL_CLASSES:
        sc = oof if cls == TRAIN_CLASS else eval_scores[(eval_df['Class'] == cls).values]
        summary.append({
            'Набор признаков': set_name, 'Class': cls, 'n': len(sc),
            'Аномалий': int((sc < threshold).sum()),
            'Доля аномалий, %': round((sc < threshold).mean() * 100, 1),
            'Медиана score': round(float(np.median(sc)), 4),
            'Порог': round(float(threshold), 4),
        })
    baikal_sc = eval_scores[(eval_df['Class'] == TEST_CLASS).values]
    auc = roc_auc_score(np.r_[np.zeros(len(oof)), np.ones(len(baikal_sc))], -np.r_[oof, baikal_sc])
    summary.append({'Набор признаков': set_name, 'Class': f'ROC AUC {TRAIN_CLASS} vs {TEST_CLASS}',
                    'n': np.nan, 'Аномалий': np.nan, 'Доля аномалий, %': np.nan,
                    'Медиана score': round(auc, 3), 'Порог': np.nan})

results_df = pd.DataFrame(results)
results_df['Score'] = results_df['Score'].round(4)
results_df['Перцентиль среди ADOM, %'] = results_df['Перцентиль среди ADOM, %'].round(1)
summary_df = pd.DataFrame(summary)

pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 80)
print(summary_df.to_string(index=False))
print()
non_train = results_df[results_df['Роль'] != 'обучение (out-of-fold)']
print(non_train.drop(columns=['Роль']).to_string(index=False))

# ---------------------------------------------------------------------------
# График: распределение скоров
# ---------------------------------------------------------------------------
colors = {TRAIN_CLASS: '#9aa5b1', TEST_CLASS: '#1f77b4', 'Tech': '#d62728', 'Osinovka': '#2ca02c'}
titles = {'all': 'Все признаки', 'intensive': 'Признаки, не зависящие от концентрации'}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150, sharey=False)
rng = np.random.default_rng(0)
order = [TRAIN_CLASS, TEST_CLASS] + EXTERNAL_CLASSES
for ax, set_name in zip(axes, FEATURE_SETS):
    oof, thr = oof_store[set_name]
    sub = results_df[results_df['Набор признаков'] == set_name]
    for i, cls in enumerate(order):
        s = sub.loc[sub['Class'] == cls, 'Score'].values
        if cls == TRAIN_CLASS:
            ax.boxplot(s, positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color='black'))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(s)), s, s=28, alpha=0.8,
                   color=colors[cls], edgecolor='white', linewidth=0.5, zorder=3)
    ax.axhline(thr, color='black', ls='--', lw=1)
    ax.text(len(order) - 0.5, thr, f' порог (5% ADOM)', va='bottom', ha='right', fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f'{c}\n(OOF)' if c == TRAIN_CLASS else c for c in order])
    ax.set_title(titles[set_name])
    ax.set_ylabel('Isolation Forest score (ниже = аномальнее)')
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plot_path = OUT_DIR / 'IsolationForest_ADOM_scores.png'
plt.savefig(plot_path)
plt.close()

# ---------------------------------------------------------------------------
# Сохранение
# ---------------------------------------------------------------------------
xlsx_path = OUT_DIR / 'IsolationForest_ADOM.xlsx'
with pd.ExcelWriter(xlsx_path) as writer:
    summary_df.to_excel(writer, sheet_name='Сводка', index=False)
    for set_name in FEATURE_SETS:
        (results_df[results_df['Набор признаков'] == set_name]
         .sort_values(['Роль', 'Score'], ascending=[False, True])
         .to_excel(writer, sheet_name=f'Образцы ({set_name})', index=False))
    pd.DataFrame({'Набор признаков': list(FEATURE_SETS),
                  'Признаки': [', '.join(v) for v in FEATURE_SETS.values()]}) \
        .to_excel(writer, sheet_name='Признаки', index=False)

print(f'\n✅ Таблица: {xlsx_path}\n✅ График: {plot_path}')