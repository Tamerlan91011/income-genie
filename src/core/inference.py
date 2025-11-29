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
    # Здесь нужно применить ту же предобработку, что и для train данных
    # Для простоты предполагаем, что new_data уже в том же формате

    # Если new_data - это путь к файлу, загружаем данные
    if isinstance(new_data, str):
        new_data = pd.read_csv(
            new_data,
            decimal=",",
            sep=";",
            engine="python",
            on_bad_lines="warn",
            encoding="UTF-8",
        )

    # Масштабирование
    new_data_scaled = scaler.transform(new_data)

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
    results = pd.DataFrame({"prediction": ensemble_prediction})

    # Добавляем предсказания отдельных моделей
    for model_name, pred in individual_predictions.items():
        results[f"{model_name}_pred"] = pred

    print(f"Сделаны предсказания для {len(new_data)} образцов")

    return results, individual_predictions


# Запуск главной функции
if __name__ == "__main__":
    inference("src/datasets/hackathon_income_test.csv")
