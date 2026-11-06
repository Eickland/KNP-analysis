import pandas as pd
from pathlib import Path
import sys
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_classification, random_forest_classification
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, CLASSIFICATION_PATH

GrindBestCombEnabled = False
SingleModelEnabled = True

# Переводить ли коэффициенты LDA в исходные единицы признаков.
# True  -> функцию можно сразу применять к "сырым" значениям Asm_280, Component_2 и т.д.
# False -> коэффициенты для стандартизованных признаков (z = (x - mean) / std), как внутри модели
LDA_RAW_UNITS = True

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH, "KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon', 'sample_subclass', 'sample_class'], min_coverage=0.5)
data.dropna(inplace=True)
data = pd.read_excel(r"C:\Users\Kirill\Desktop\Lab\KNP-analysis\Received_data\Statistics\2024\Documents\Исходные данные за 2024 год.xlsx")
data = data[data['Class'] != 'Tech']
data = data[data['Class'] != 'Osinovka']
# Создаем маски для разных условий
#mask_adom = (MainData['sample_class'] == 'ADOM')
MainData = ata
MainData.set_index(["Class", "Subclass", "Sample"], inplace=True)


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ СВОДНОЙ ТАБЛИЦЫ
# ============================================================================

METRIC_COLUMNS = {
    'precision': 'Точность (precision)',
    'recall': 'Полнота (recall)',
    'f1-score': 'F1-мера',
}

SUMMARY_COLUMNS = (['Признаки', 'Класс образцов', 'К-во образцов', 'Размер тестовой выборки']
                   + list(METRIC_COLUMNS.values())
                   + ['Accuracy модели'])


def _report_row(clf_report_df: pd.DataFrame, keys) -> dict:
    """
    Метрики одной строки отчета классификации.
    В LDA строки отчета подписаны именами классов, в RF - номерами ('0', '1', ...),
    поэтому перебираем несколько вариантов ключа. Если строки нет
    (класс не попал в тестовую выборку) - NaN.
    """
    for key in keys:
        if key in clf_report_df.index:
            return {col: float(clf_report_df.loc[key, m]) for m, col in METRIC_COLUMNS.items()}
    return {col: np.nan for col in METRIC_COLUMNS.values()}


def _metric_rows(features_str, class_names, clf_report_df, y_train, y_test, accuracy) -> list:
    """
    Строки сводной таблицы: по одной на класс + macro avg + weighted avg.
    Accuracy - общая для модели, повторяется во всех ее строках (удобно фильтровать в Excel).
    """
    n_classes = len(class_names)
    counts_all = _class_counts(y_train, y_test, n_classes)
    counts_test = np.bincount(np.asarray(y_test), minlength=n_classes)

    rows = []
    for idx, cls in enumerate(class_names):
        rows.append({
            'Признаки': features_str,
            'Класс образцов': cls,
            'К-во образцов': int(counts_all[idx]),
            'Размер тестовой выборки': int(counts_test[idx]),
            **_report_row(clf_report_df, (cls, str(cls), str(idx), idx)),
            'Accuracy модели': accuracy,
        })

    for key, label in (('macro avg', 'Среднее (macro avg)'),
                       ('weighted avg', 'Среднее (weighted avg)')):
        rows.append({
            'Признаки': features_str,
            'Класс образцов': label,
            'К-во образцов': int(counts_all.sum()),
            'Размер тестовой выборки': int(counts_test.sum()),
            **_report_row(clf_report_df, (key,)),
            'Accuracy модели': accuracy,
        })
    return rows


def _class_counts(y_train, y_test, n_classes: int) -> np.ndarray:
    """
    Количество образцов каждого класса во всем наборе (обучение + тест),
    который ушел в модель, т.е. после фильтрации IQR и удаления пропусков.
    """
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
    return np.bincount(y_all, minlength=n_classes)


def _format_linear(coefs, intercept, feature_names) -> str:
    """Строка вида '+0.1234·Asm_280 -5.678·Component_2 +1.234'"""
    parts = [f"{c:+.4g}·{f}" for c, f in zip(coefs, feature_names)]
    parts.append(f"{intercept:+.4g}")
    return " ".join(parts)


def _lda_functions(lda_model, scaler, feature_names, class_names, raw_units=True) -> dict:
    """
    Дискриминантные функции LDA для каждого класса.

    Многоклассовый случай: coef_ имеет форму (n_classes, n_features), у каждого класса
    своя функция δ_k(x); объект относится к классу с наибольшим δ_k(x).

    Бинарный случай: coef_ имеет форму (1, n_features) - это одна функция D(x)
    (разность дискриминантов). D(x) > 0 -> второй класс, D(x) < 0 -> первый.
    """
    coef = np.atleast_2d(lda_model.coef_)
    intercept = np.atleast_1d(lda_model.intercept_)

    if raw_units:
        # w·z + b, где z = (x - mean)/std  ->  (w/std)·x + (b - Σ w·mean/std)
        coef_out = coef / scaler.scale_
        intercept_out = intercept - (coef * scaler.mean_ / scaler.scale_).sum(axis=1)
    else:
        coef_out, intercept_out = coef, intercept

    functions = {}
    if coef.shape[0] == 1 and len(class_names) == 2:
        expr = _format_linear(coef_out[0], intercept_out[0], feature_names)
        functions[class_names[0]] = f"D(x) = {expr};  D(x) < 0 → {class_names[0]}"
        functions[class_names[1]] = f"D(x) = {expr};  D(x) > 0 → {class_names[1]}"
    else:
        for i, cls in enumerate(class_names):
            functions[cls] = f"δ_{cls}(x) = {_format_linear(coef_out[i], intercept_out[i], feature_names)}"
    return functions


def _feature_order(df: pd.DataFrame) -> list:
    """
    Порядок признаков, в котором их видела модель (числовые колонки в исходном порядке).
    Брать его из feature_importance нельзя - там признаки отсортированы по важности.
    """
    return list(df.select_dtypes(include=[np.number]).columns)


def collect_lda_rows(res_lda, feature_comb, raw_units=True) -> list:
    (results_df, lda_model, scaler, feature_importance, label_encoder,
     X_train, X_test, y_train, y_test, X_train_lda, X_test_lda,
     accuracy, weighted_avg_accuracy, macro_avg_accuracy,
     clf_report_df, external_results) = res_lda

    class_names = list(label_encoder.classes_)
    feature_names = _feature_order(MainData[feature_comb])
    functions = _lda_functions(lda_model, scaler, feature_names, class_names, raw_units)

    rows = _metric_rows(", ".join(feature_comb), class_names, clf_report_df, y_train, y_test, accuracy)
    for row in rows:
        row['Функция'] = functions.get(row['Класс образцов'], '')
    return rows


def collect_lda_coordinates(res_lda, feature_comb) -> tuple:
    """
    Координаты образцов в пространстве LDA (LD1, LD2, ...).

    Координаты считаются заново для всех образцов: lda.transform(scaler.transform(X)),
    чтобы у каждой точки были Class/Subclass/Sample (в X_train_lda/X_test_lda их нет).
    Принадлежность к обучающей/тестовой выборке определяется сопоставлением
    с X_test_lda, поэтому не зависит от того, в каком виде функция вернула X_train/X_test.

    Возвращает (таблица координат, таблица объясненной дисперсии).
    """
    (results_df, lda_model, scaler, feature_importance, label_encoder,
     X_train, X_test, y_train, y_test, X_train_lda, X_test_lda,
     accuracy, weighted_avg_accuracy, macro_avg_accuracy,
     clf_report_df, external_results) = res_lda

    df = MainData[feature_comb].dropna()
    feature_names = _feature_order(df)
    X_scaled = scaler.transform(df[feature_names].values)
    coords = lda_model.transform(X_scaled)
    ld_cols = [f'LD{i + 1}' for i in range(coords.shape[1])]

    out = df.index.to_frame(index=False)
    out[ld_cols] = coords

    # Обучение / тест: точка считается тестовой, если ее координаты совпадают с одной из X_test_lda
    test_coords = np.asarray(X_test_lda)
    if test_coords.ndim == 1:
        test_coords = test_coords.reshape(-1, 1)
    test_coords = test_coords[:, :coords.shape[1]]
    is_test = np.zeros(len(coords), dtype=bool)
    for tc in test_coords:
        d = np.abs(coords - tc).max(axis=1)
        is_test[np.argmin(d)] |= d.min() < 1e-6
    out['Выборка'] = np.where(is_test, 'тест', 'обучение')

    # Предсказанный класс и вероятности
    out['Предсказанный класс'] = label_encoder.inverse_transform(lda_model.predict(X_scaled))
    proba = lda_model.predict_proba(X_scaled)
    for i, cls in enumerate(label_encoder.classes_):
        out[f'P({cls})'] = proba[:, i].round(4)
    out[ld_cols] = out[ld_cols].round(5)

    # Центроиды классов в пространстве LDA
    true_col = df.index.names[0]
    centroids = out.groupby(true_col)[ld_cols].mean().round(5)
    centroids.index = [f'Центроид {c}' for c in centroids.index]

    var = pd.DataFrame({
        'Признаки': ", ".join(feature_comb),
        'Ось': ld_cols,
        'Доля объясненной дисперсии': np.round(lda_model.explained_variance_ratio_[:len(ld_cols)], 4),
    })
    return out, centroids, var


def collect_rf_rows(res_rf, feature_comb) -> list:
    (results_df, rf_model, scaler, feature_importance,
     X_train, X_test, y_train, y_test, label_encoder,
     accuracy, weighted_avg_accuracy, macro_avg_accuracy,
     clf_report_df, external_results) = res_rf

    class_names = list(label_encoder.classes_)
    return _metric_rows(", ".join(feature_comb), class_names, clf_report_df, y_train, y_test, accuracy)


# ============================================================================
# ОСНОВНОЙ РАСЧЕТ
# ============================================================================

if SingleModelEnabled:

    feature_comb_list = [
                        ['Asm_280', 'Component_2'],
                        ['Asm_280', 'Component_2', 'FIX'],
                        ['Asm_280', 'Suva', 'FIX', 'C_org'],
                        ]

    lda_rows = []
    rf_rows = []
    lda_coords = []      # (комбинация, координаты, центроиды)
    lda_variance = []

    for feature_comb in feature_comb_list:
        res_lda = lda_classification(MainData[feature_comb], index_level=0,
                                     output_html_path=str(Path.joinpath(CLASSIFICATION_PATH, "LDA", f"LDA_{feature_comb}.html")),
                                     external_validation=False,
                                     )

        res_rf = random_forest_classification(MainData[feature_comb], index_level=0,
                                              output_html_path=str(Path.joinpath(CLASSIFICATION_PATH, "RF", f"RF_{feature_comb}.html")),
                                              external_validation=False,
                                              )

        lda_rows.extend(collect_lda_rows(res_lda, feature_comb, raw_units=LDA_RAW_UNITS))
        rf_rows.extend(collect_rf_rows(res_rf, feature_comb))

        coords_df, centroids_df, var_df = collect_lda_coordinates(res_lda, feature_comb)
        lda_coords.append((feature_comb, coords_df, centroids_df))
        lda_variance.append(var_df)

    lda_summary = pd.DataFrame(lda_rows, columns=SUMMARY_COLUMNS + ['Функция'])
    rf_summary = pd.DataFrame(rf_rows, columns=SUMMARY_COLUMNS)

    metric_cols = list(METRIC_COLUMNS.values()) + ['Accuracy модели']
    lda_summary[metric_cols] = lda_summary[metric_cols].round(4)
    rf_summary[metric_cols] = rf_summary[metric_cols].round(4)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 250)
    print("\nLDA:")
    print(lda_summary.to_string(index=False))
    print("\nRandom Forest:")
    print(rf_summary.to_string(index=False))

    summary_path = Path.joinpath(CLASSIFICATION_PATH, "Models_summary.xlsx")
    with pd.ExcelWriter(summary_path) as writer:
        lda_summary.to_excel(writer, sheet_name='LDA', index=False)
        rf_summary.to_excel(writer, sheet_name='RF', index=False)

    # То же самое в CSV с разделителем ';' (откроется в русском Excel без мастера импорта)
    lda_summary.to_csv(Path.joinpath(CLASSIFICATION_PATH, "LDA_summary.csv"), sep=';', index=False, encoding='utf-8-sig')
    rf_summary.to_csv(Path.joinpath(CLASSIFICATION_PATH, "RF_summary.csv"), sep=';', index=False, encoding='utf-8-sig')

    print(f"\n✅ Сводные таблицы сохранены: {summary_path}")

    # ------------------------------------------------------------------
    # Координаты LDA - отдельный файл, по листу на комбинацию признаков
    # ------------------------------------------------------------------
    coords_path = Path.joinpath(CLASSIFICATION_PATH, "LDA_coordinates.xlsx")
    combos_index = []
    with pd.ExcelWriter(coords_path) as writer:
        for i, (feature_comb, coords_df, centroids_df) in enumerate(lda_coords, start=1):
            # Имя листа Excel: не длиннее 31 символа и без []:*?/\
            sheet = f"{i}_" + "+".join(feature_comb)
            sheet = "".join('_' if ch in '[]:*?/\\' else ch for ch in sheet)[:31]
            coords_df.to_excel(writer, sheet_name=sheet, index=False)
            # Центроиды классов - под таблицей образцов, через пустую строку
            centroids_df.to_excel(writer, sheet_name=sheet, startrow=len(coords_df) + 2)
            combos_index.append({'Лист': sheet, 'Признаки': ", ".join(feature_comb),
                                 'Число осей LD': coords_df.filter(like='LD').shape[1],
                                 'Образцов': len(coords_df)})
        pd.DataFrame(combos_index).to_excel(writer, sheet_name='Комбинации', index=False)
        pd.concat(lda_variance).to_excel(writer, sheet_name='Объясненная дисперсия', index=False)

    print(f"✅ Координаты LDA сохранены: {coords_path}")

#model_data = pd.read_excel(Path.joinpath(Path.cwd(),"Server","Received_data","Model",f"Model_accuracy_cross-val.xlsx"))