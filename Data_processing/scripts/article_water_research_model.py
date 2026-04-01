import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
import HumSpectra.fluorescence as fl
import HumSpectra.ultraviolet as uv 
import HumSpectra.utilits as ut
from HumSpectra.hum_statistic import lda_classification, random_forest_classification,lda_classification_batch,random_forest_classification_batch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import OneClassSVM

project_root = Path.cwd()
sys.path.insert(0, str(project_root))
plt.rcParams['axes.titlesize'] = 24

plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['axes.labelsize'] = 20

from HumSpectra.sample import SampleCollection
from Data_processing.Utilits.config_paths import CACHE_DATA_PATH, CLASSIFICATION_PATH

def train_ocsvm_detector(train_data, test_sludge, test_baikal, features, nu=0.05, kernel='rbf', gamma='scale'):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[features])
    X_test_sludge = scaler.transform(test_sludge[features])
    X_test_baikal = scaler.transform(test_baikal[features])
    
    model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma) # type: ignore
    model.fit(X_train)
    
    # Получаем оценки расстояния до границы
    dist_sludge = model.decision_function(X_test_sludge)
    dist_baikal = model.decision_function(X_test_baikal)
    
    pred_sludge = model.predict(X_test_sludge)
    pred_baikal = model.predict(X_test_baikal)
    
    y_true = [1] * len(pred_sludge) + [-1] * len(pred_baikal)
    y_pred = list(pred_sludge) + list(pred_baikal)
    
    cl_report = classification_report(y_true, y_pred, output_dict=True)
    
    return model, scaler, cl_report, dist_sludge, dist_baikal

def train_anomaly_detector(train_data, test_sludge, test_baikal, features):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data[features])
    X_test_sludge = scaler.transform(test_sludge[features])
    X_test_baikal = scaler.transform(test_baikal[features])
    
    model = IsolationForest(n_estimators=100, contamination=0.13, random_state=42)
    model.fit(X_train)
    
    # Оценки аномальности (чем меньше, тем более аномально)
    score_sludge = model.decision_function(X_test_sludge)
    score_baikal = model.decision_function(X_test_baikal)
    
    pred_sludge = model.predict(X_test_sludge)
    pred_baikal = model.predict(X_test_baikal)
    
    y_true = [1] * len(pred_sludge) + [-1] * len(pred_baikal)
    y_pred = list(pred_sludge) + list(pred_baikal)
    
    cl_report = classification_report(y_true, y_pred, output_dict=True)
    
    return model, scaler, cl_report, score_sludge, score_baikal

def plot_anomaly_distribution(score_sludge, score_baikal, model_name, save_path=None):
    """Визуализация распределения оценок аномальности для двух групп"""
    plt.figure(figsize=(10, 6))
    
    sns.histplot(score_sludge, kde=True, color='blue', label='Шлам (Тест)', stat="density", alpha=0.5)
    sns.histplot(score_baikal, kde=True, color='red', label='Байкал (Аномалия)', stat="density", alpha=0.5)
    
    plt.axvline(0, color='black', linestyle='--', label='Граница раздела')
    plt.title(f'Распределение оценок уверенности: {model_name}')
    plt.xlabel('Score (Decision Function)')
    plt.ylabel('Плотность')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

yandex_disk_path = os.getenv('YANDEX_DISK_PATH')
if yandex_disk_path is None:
    raise KeyError('Ошибка подключения яндекс диска')

collection = SampleCollection(cache_dir=CACHE_DATA_PATH)
collection.load_collection(str(Path.joinpath(CACHE_DATA_PATH,"KNP_samples.pkl")))
print(f'Коллекция загружена, к-во образцов: {len(collection.samples)}')

data = collection.create_flexible_descriptors_table(sample_attributes=['org_carbon','sample_subclass','sample_class'],min_coverage=0.25)
data.dropna(inplace=True)
print(data.columns)
MainData = ut.filter_by_iqr_per_group(data,iqr_param=2,target_column='Subclass', filter_columns=["asm_350",'org_carbon'])
MainData = MainData[MainData['Subclass'] != 'Storage 8']
#MainData.set_index(["Class","Subclass","Sample"],inplace=True)

MultiSearchOn = False
SearchOn = False
"""
if MultiSearchOn:
    feature_comb = ['Asm_280','org_carbon','Component_1','Component_2','Component_3','Asm_350','HIX','FIX']
    num_fearure_list = [1,2,3,4,5,6,7,8]
    num_iterations = 0
    k = 1


    accuracy_lda = []
    weight_accuracy_lda=[]
    accuracy_rf = []
    weight_accuracy_rf=[]
    combinations_2Dlist = []
    count_feature_list = []        

    for num_fearure_max in num_fearure_list:
        num_iterations += len(list(combinations(feature_comb, num_fearure_max)))

    for num_fearure_max in num_fearure_list:

        combinations_list = list(combinations(feature_comb, num_fearure_max))

        for iteration in combinations_list:
            
            print(f'{k}/{num_iterations}')
            
            # Извлекаем имя для файла (без скобок и кавычек)
            if isinstance(iteration, tuple):
                if len(iteration) == 1:
                    file_name = str(iteration[0])  # 'Ag_380'
                else:
                    # Если несколько признаков, объединяем их
                    file_name = "_".join([str(f) for f in iteration])  # 'Ag_380_asm_350'
            else:
                file_name = str(iteration)
            
            # Исправленный путь для LDA
            lda_path = Path(yandex_disk_path) / "article_water_research" / "models" / "lda" / f"{file_name}.html"
            res_lda = lda_classification_batch(
                MainData[list(iteration)],
                index_level=0,
                output_html_path=None,
                external_validation=False
            )
            weight_accuracy_lda.append(res_lda["weighted_avg_accuracy"])
            accuracy_lda.append(res_lda["accuracy"])        
            
            # Исправленный путь для RF
            rf_path = Path(yandex_disk_path) / "article_water_research" / "models" / "rf" / f"{file_name}.html"
            res_rf = random_forest_classification_batch(
                MainData[list(iteration)],
                index_level=0,
                output_html_path=None,
                external_validation=False
            )
            
            weight_accuracy_rf.append(res_rf["weighted_avg_accuracy"])
            accuracy_rf.append(res_rf["accuracy"])
            
            combinations_2Dlist.append(iteration)
            count_feature_list.append(num_fearure_max)
            k += 1
            
    accuracy_data = pd.DataFrame()

    accuracy_data['LDA_weight_accuracy'] = weight_accuracy_lda
    accuracy_data['LDA_accuracy'] = accuracy_lda    
    accuracy_data['RF_weight_accuracy'] = weight_accuracy_rf
    accuracy_data['RF_accuracy'] = accuracy_rf
    accuracy_data['feature'] = combinations_2Dlist
    accuracy_data['num_feature'] = count_feature_list

    accuracy_data.sort_values('LDA_weight_accuracy',ascending=False).to_excel(str(Path(yandex_disk_path) / "article_water_research" / "models" / "models_data.xlsx"))
"""
if SearchOn:
    
    feature_comb = ['Asm_280','org_carbon','Component_1','Component_2','Component_3','Asm_350','HIX','FIX','Suva','Ag_275','Ag_380','Ir_19_20','E2E3','B1','B2']
    num_fearure_list = [1,2,3,4,5]
    num_iterations = 0
    k = 1


    baikal_recall_list = []
    sludge_recall_list = []
    oneclasssvm_list = []
    combinations_2Dlist = []
    count_feature_list = []
    oneclass_dist_b_list = []        

    for num_fearure_max in num_fearure_list:
        all_combs = list(combinations(feature_comb, num_fearure_max))
        # Оставляем только комбинации с 'org_carbon'
        filtered_combs = [comb for comb in all_combs if 'org_carbon' in comb]
        num_iterations += len(filtered_combs)

    for num_fearure_max in num_fearure_list:
        combinations_list = list(combinations(feature_comb, num_fearure_max))
        # Фильтруем комбинации
        filtered_combinations_list = [comb for comb in combinations_list if 'org_carbon' in comb]
        

        for iteration in filtered_combinations_list:
            
            print(f'{k}/{num_iterations}')
            
            # Извлекаем имя для файла (без скобок и кавычек)
            if isinstance(iteration, tuple):
                if len(iteration) == 1:
                    file_name = str(iteration[0])  # 'Ag_380'
                else:
                    # Если несколько признаков, объединяем их
                    file_name = "_".join([str(f) for f in iteration])  # 'Ag_380_asm_350'
            else:
                file_name = str(iteration)
                    
            sludge_all = MainData[MainData['Class'] == 'ADOM']
            baikal_all = MainData[MainData['Class'] == 'Baikal']

            # Разделяем шлам на обучение и тест (80/20)
            train_sludge = sludge_all.sample(frac=0.8, random_state=42)
            test_sludge = sludge_all.drop(train_sludge.index)
            
            # Внутри цикла combinations_list:
            model_if, scaler_if, res_if, dist_s_if, dist_b_if = train_anomaly_detector(
                train_data=train_sludge,
                test_sludge=test_sludge,
                test_baikal=baikal_all,
                features=list(iteration)
            )
                
            baikal_recall_list.append(res_if.get('-1', res_if.get('Байкал (Чужой)', {})).get('recall', 0)) # type: ignore
            sludge_recall_list.append(res_if.get('1', res_if.get('Шлам (Свой)', {})).get('recall', 0)) # type: ignore
            
            model_svm, scaler_svm, res_svm, dist_s_svm, dist_b_svm = train_ocsvm_detector(
                train_data=train_sludge,
                test_sludge=test_sludge,
                test_baikal=baikal_all,
                features=list(iteration)
            )   
            
            oneclasssvm_list.append(res_svm['macro avg']['f1-score']) # type: ignore # Или среднее dist_b_svm.mean()
            oneclass_dist_b_list.append(dist_b_svm)            
            combinations_2Dlist.append(iteration)
            count_feature_list.append(num_fearure_max)
            k += 1
            
    accuracy_data = pd.DataFrame()
    accuracy_data['Признаки'] = combinations_2Dlist
    accuracy_data['К-во признаков'] = count_feature_list
    accuracy_data['Чувствительность Бакала'] = baikal_recall_list
    accuracy_data['Чувствительность шлама'] = sludge_recall_list    
    accuracy_data['One class SVM, f1-score'] = oneclasssvm_list
    accuracy_data['Средняя чужеродность'] = oneclass_dist_b_list
            
    accuracy_data.sort_values('Чувствительность Бакала',ascending=False).to_excel(str(Path(yandex_disk_path) / "article_water_research" / "models" / "detect_data.xlsx"))
    
choosed_combinations = ['org_carbon']

def analyze_carbon_sensitivity(MainData, features, feature_to_shift='org_carbon', steps=50):
    print(f"Начинаем анализ чувствительности для признака: {feature_to_shift}")
    
    # 1. Подготовка данных
    sludge_all = MainData[MainData['Class'] == 'ADOM']
    baikal_all = MainData[MainData['Class'] == 'Baikal']
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_sludge = scaler.fit_transform(sludge_all[features])
    
    # Вычисляем физический предел сдвига (разница средних + 50% запаса)
    mean_sludge = sludge_all[feature_to_shift].mean()
    mean_baikal = baikal_all[feature_to_shift].mean()
    max_shift = abs(mean_sludge - mean_baikal) * 1.5 
    print(f"Среднее в шламе: {mean_sludge:.2f}, в Байкале: {mean_baikal:.2f}. Макс. сдвиг: {max_shift:.2f}")

    # 2. Обучение эталонных моделей
    # IF
    iforest = IsolationForest(n_estimators=100, contamination=0.12, random_state=42)
    iforest.fit(X_train_sludge)
    
    # OCSVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='scale')
    ocsvm.fit(X_train_sludge)
    
    # 3. Цикл постепенного сдвига признака
    shift_values = np.linspace(0, max_shift, steps)
    results = []
    
    for shift in shift_values:
        # Создаем копию Байкала и увеличиваем значение
        baikal_shifted = baikal_all.copy()
        baikal_shifted[feature_to_shift] += shift
        
        # Масштабируем через уже ОБУЧЕННЫЙ scaler
        X_baikal_shifted = scaler.transform(baikal_shifted[features])
        
        # Предсказания (Доля -1 — это наш Recall для Байкала)
        pred_if = iforest.predict(X_baikal_shifted)
        recall_if = (pred_if == -1).mean()
        dist_if = iforest.decision_function(X_baikal_shifted).mean()
        
        pred_svm = ocsvm.predict(X_baikal_shifted)
        recall_svm = (pred_svm == -1).mean()
        dist_svm = ocsvm.decision_function(X_baikal_shifted).mean()
        
        results.append({
            'Shift': shift,
            'Real_Carbon_Value': mean_baikal + shift,
            'IF_Recall': recall_if,
            'IF_Distance': dist_if,
            'SVM_Recall': recall_svm,
            'SVM_Distance': dist_svm
        })
        
    df_res = pd.DataFrame(results)
    
    # ==========================================
    # 4. КРАСИВАЯ ВИЗУАЛИЗАЦИЯ
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Анализ чувствительности: Влияние роста {feature_to_shift} на распознавание Байкала', fontsize=16)

    # График 1: Падение Recall (Способность видеть разницу)
    ax1.plot(df_res['Real_Carbon_Value'], df_res['IF_Recall'], label='Isolation Forest', color='#2ca02c', linewidth=2.5)
    ax1.plot(df_res['Real_Carbon_Value'], df_res['SVM_Recall'], label='One-Class SVM', color='#1f77b4', linewidth=2.5)
    
    ax1.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Случайное угадывание (0.5)')
    ax1.axvline(mean_sludge, color='grey', linestyle=':', label='Среднее значение шлама')
    
    ax1.set_title('Деградация чувствительности (Recall)', fontsize=20)
    ax1.set_xlabel(f'Абсолютное значение {feature_to_shift} в Байкале', fontsize=20)
    ax1.set_ylabel('Доля распознанных аномалий (Recall)', fontsize=20)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='lower left')

    # График 2: Приближение к границе нормы (Decision Function)
    ax2.plot(df_res['Real_Carbon_Value'], df_res['IF_Distance'], label='Isolation Forest Dist.', color='#2ca02c', linestyle='-.', linewidth=2)
    ax2.plot(df_res['Real_Carbon_Value'], df_res['SVM_Distance'], label='One-Class SVM Dist.', color='#1f77b4', linestyle='-.', linewidth=2)
    
    ax2.axhline(0, color='black', linestyle='-', alpha=0.8, label='Граница нормы (0.0)')
    ax2.axvline(mean_sludge, color='grey', linestyle=':', label='Среднее значение шлама')
    
    ax2.set_title('Приближение к эталонной сигнатуре', fontsize=16)
    ax2.set_xlabel(f'Абсолютное значение {feature_to_shift} в Байкале', fontsize=20)
    ax2.set_ylabel('Оценка "нормальности" (<0 = аномалия, >0 = шлам)', fontsize=20)
    ax2.legend(loc='upper left')

    plt.tight_layout()
    
    # Сохраняем график
    save_path = str(Path(yandex_disk_path) / "article_water_research" / "models" / f"sensitivity_{feature_to_shift}.png") # type: ignore
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен по пути: {save_path}")
    plt.show()

    return df_res

# ==========================================
# ЗАПУСК (Вставьте этот блок после вашего SearchOn)
# ==========================================


# Запускаем анализ
sensitivity_df = analyze_carbon_sensitivity(
    MainData=MainData, 
    features=choosed_combinations, 
    feature_to_shift='org_carbon', 
    steps=60
)                        