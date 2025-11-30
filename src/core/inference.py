import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
import pickle
import os
from typing import Dict, Tuple, Optional, Union
from scipy.sparse import spmatrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def load_models_and_artifacts(
    model_dir: str = "saved_models", version: Optional[str] = None
) -> Tuple[
    Dict[
        str,
        Union[CatBoostRegressor, RandomForestRegressor, XGBRegressor, LGBMRegressor],
    ],
    StandardScaler,
    np.ndarray,
    pd.DataFrame,
    Optional[str],
]:
    """Загрузка моделей и артефактов"""

    print(f"\nЗАГРУЗКА МОДЕЛЕЙ И АРТЕФАКТОВ ИЗ ПАПКИ '{model_dir}'...")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Папка {model_dir} не существует")

    # Если версия не указана, загружаем последнюю
    if version is None:
        files = [f for f in os.listdir(model_dir) if f.startswith("model_info_")]
        if not files:
            raise FileNotFoundError("Не найдены файлы с моделями")

        # Берем последнюю версию
        versions = [f.replace("model_info_", "").replace(".txt", "") for f in files]
        versions.sort(reverse=True)
        version = versions[0]
        print(f"Загружаем последнюю версию: {version}")

    models: Dict[
        str,
        Union[CatBoostRegressor, RandomForestRegressor, XGBRegressor, LGBMRegressor],
    ] = {}

    # Загружаем модели
    model_files = [
        f
        for f in os.listdir(model_dir)
        if version in f
        and f.endswith(".pkl")
        and not f.startswith("scaler")
        and not f.startswith("ensemble_weights")
    ]

    for model_file in model_files:
        model_name = model_file.replace(f"_{version}.pkl", "")
        model_path = os.path.join(model_dir, model_file)

        with open(model_path, "rb") as f:
            models[model_name] = pickle.load(f)

        print(f"Загружена модель: {model_name}")

    # Загружаем scaler
    scaler_path = os.path.join(model_dir, f"scaler_{version}.pkl")
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    print(f"Загружен scaler: {scaler_path} ({type(scaler)})")

    # Загружаем веса ансамбля
    weights_path = os.path.join(model_dir, f"ensemble_weights_{version}.pkl")
    with open(weights_path, "rb") as f:
        ensemble_weights: np.ndarray = pickle.load(f)
    print(f"Загружены веса ансамбля: {weights_path} ({type(ensemble_weights)})")

    # Загружаем feature importance
    feature_importance_path = os.path.join(
        model_dir, f"feature_importance_{version}.csv"
    )
    feature_importance = pd.read_csv(feature_importance_path)
    print(f"Загружена важность признаков: {feature_importance_path})")

    return models, scaler, ensemble_weights, feature_importance, version


def preprocess_data(df_test):
    """Комплексная предобработка данных для test/inference"""

    print("Начало предобработки...")

    # Копируем данные
    test_processed = df_test.copy()

    # Удаление служебных колонок


    X_test = test_processed

    # Разделение на числовые и категориальные
    numeric_features = X_test.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_test.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Числовых признаков: {len(numeric_features)}")
    print(f"Категориальных признаков: {len(categorical_features)}")

    # Обработка категориальных признаков
    if categorical_features:
        print("Кодирование категориальных признаков...")
        for col in categorical_features:
            if col in X_test.columns:
                # Заполняем пропуски перед кодированием
                X_test[col] = X_test[col].fillna("missing")
                # В реальной ситуации нам бы понадобились сохраненные LabelEncoder объекты
                # Но для демонстрации просто кодируем как числа
                X_test[col] = pd.Categorical(X_test[col]).codes

    # Импутация пропусков для числовых признаков
    if numeric_features:
        print("Импутация пропусков для числовых признаков...")
        # Note: In inference, we'll need to use the same imputer that was fitted during training
        # For now, we'll just fill with median as a placeholder
        for col in numeric_features:
            if col in X_test.columns:
                X_test[col] = X_test[col].fillna(X_test[col].median())

    print(f"Финальные размеры: X_test {X_test.shape}")

    return X_test


def create_features(X_test):
    """Создание дополнительных признаков для test/inference"""

    X_test_fe = X_test.copy()

    print("Создание новых признаков...")

    # Агрегированные признаки по группам

    # 1. Соотношение кредитовых и дебетовых оборотов
    if (
        "turn_cur_cr_avg_v2" in X_test_fe.columns
        and "turn_cur_db_avg_v2" in X_test_fe.columns
    ):
        X_test_fe["credit_debit_ratio"] = X_test_fe["turn_cur_cr_avg_v2"] / (
            X_test_fe["turn_cur_db_avg_v2"] + 1
        )

    # 2. Доля остатков от оборотов
    if (
        "curr_rur_amt_cm_avg" in X_test_fe.columns
        and "turn_cur_db_avg_v2" in X_test_fe.columns
    ):
        X_test_fe["balance_to_turnover"] = X_test_fe["curr_rur_amt_cm_avg"] / (
            X_test_fe["turn_cur_db_avg_v2"] + 1
        )

    # 3. Интенсивность использования кредитных продуктов
    if "hdb_bki_total_products" in X_test_fe.columns and "age" in X_test_fe.columns:
        X_test_fe["products_per_age"] = X_test_fe["hdb_bki_total_products"] / (
            X_test_fe["age"] + 1
        )

    # 4. Долговая нагрузка
    if (
        "hdb_outstand_sum" in X_test_fe.columns
        and "salary_6to12m_avg" in X_test_fe.columns
    ):
        X_test_fe["debt_to_income"] = X_test_fe["hdb_outstand_sum"] / (
            X_test_fe["salary_6to12m_avg"] + 1
        )

    # 5. Средний оборот по транзакции
    transaction_cols_test = [
        col for col in X_test_fe.columns if "by_category" in col or "transaction" in col
    ]
    if transaction_cols_test:
        X_test_fe["avg_transaction_amount"] = X_test_fe[transaction_cols_test].mean(
            axis=1
        )

    print(f"Создано новых признаков: {len(X_test_fe.columns) - len(X_test.columns)}")

    return X_test_fe


def select_features(X_test, feature_importance, n_features=100):
    """Отбор признаков на основе сохраненной важности признаков"""
    
    print(f"Отбор топ-{n_features} признаков...")
    
    # Выбираем топ-N признаков из сохраненной важности
    top_features = feature_importance.head(n_features)["feature"].tolist()
    
    # Оставляем только те признаки, которые есть в данных
    available_features = [feat for feat in top_features if feat in X_test.columns]
    
    # Применяем отбор
    X_test_selected = X_test[available_features]
    
    # Добавляем недостающие признаки с нулевыми значениями, чтобы соответствовать обучению
    missing_features = [feat for feat in top_features if feat not in X_test.columns]
    for feat in missing_features:
        X_test_selected[feat] = 0
    
    # Упорядочиваем столбцы в том же порядке, что и при обучении
    X_test_selected = X_test_selected[top_features]
    
    print(f"После отбора признаков: X_test {X_test_selected.shape}")
    
    return X_test_selected


# ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЙ НА НОВЫХ ДАННЫХ


def inference(
    new_data: pd.DataFrame | np.ndarray | str,
    model_dir: str = "saved_models",
    version: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray | list | spmatrix]]:
    """Функция для предсказаний на новых данных с использованием сохраненных моделей"""

    print("ЗАГРУЗКА МОДЕЛЕЙ ДЛЯ ПРЕДСКАЗАНИЙ...")

    # Загружаем модели и артефакты
    models, scaler, ensemble_weights, feature_importance, loaded_version = (
        load_models_and_artifacts(model_dir, version)
    )

    print(f"Используется версия моделей: {loaded_version}")

    # Предобработка новых данных (аналогично train)
    # Если new_data - это путь к файлу, загружаем данные
    if isinstance(new_data, str):
        new_data = pd.read_csv(
            new_data,
            decimal=",",
            sep=";",
            engine="python",
            on_bad_lines="warn",
            encoding="UTF-8"
        )
    
    # Применяем ту же предобработку, что и для train данных
    print("Применение предобработки данных...")
    new_data_processed = preprocess_data(new_data)
    
    test_ids = new_data_processed["id"] if "id" in new_data_processed.columns else None # type: ignore
    
    print("Создание дополнительных признаков...")
    new_data_features = create_features(new_data_processed)
    
    print("Отбор признаков...")
    new_data_selected = select_features(new_data_features, feature_importance)

    # Масштабирование
    # Note: We're using a simplified approach here to avoid feature mismatch issues
    # In a real-world scenario, we would need to save the preprocessing parameters
    try:
        new_data_scaled = scaler.transform(new_data_selected)
    except ValueError as e:
        print(f"Ошибка масштабирования: {e}")
        print("Используем упрощенный подход для демонстрации")
        # Просто используем данные как есть (без масштабирования)
        new_data_scaled = new_data_selected.values

    # Предсказания отдельных моделей
    individual_predictions = {}
    for model_name, model in models.items():
        individual_predictions[model_name] = model.predict(new_data_scaled)

    # Ансамблевое предсказание (взвешенное)
    ensemble_prediction = np.average(
        [pred for pred in individual_predictions.values()],
        axis=0,
        weights=ensemble_weights,
    )

    # Создаем DataFrame с результатами
    results = pd.DataFrame({"id": test_ids, "prediction": ensemble_prediction})

    # Добавляем предсказания отдельных моделей
    for model_name, pred in individual_predictions.items():
        results[f"{model_name}_pred"] = pred

    print(f"Сделаны предсказания для {len(new_data)} образцов")

    return results, individual_predictions


# Запуск главной функции
if __name__ == "__main__":
    inference("src/datasets/hackathon_income_test.csv")
