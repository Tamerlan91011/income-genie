import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Загрузка данных
df_train = pd.read_csv(
    "src/datasets/hackathon_income_train.csv",
    decimal=",",
    sep=";",
    engine="python",
    on_bad_lines="warn",
    encoding="UTF-8",
)
df_test = pd.read_csv(
    "src/datasets/hackathon_income_test.csv",
    decimal=",",
    sep=";",
    engine="python",
    on_bad_lines="warn",
    encoding="UTF-8",
)

print("Размер train датасета:", df_train.shape)
print("Размер test датасета:", df_test.shape)


def weighted_mae(y_true, y_pred, weights):
    """Weighted Mean Absolute Error (WMAE)"""
    return np.average(np.abs(y_true - y_pred), weights=weights)


def calculate_wmae_by_income_group(
    y_true, y_pred, y_true_values, group_bins=[0, 50000, 100000, 200000, 500000, np.inf]
):
    """Вычисление WMAE по группам доходов"""
    results = {}
    group_labels = [
        f"до {group_bins[i + 1]:,.0f}"
        if i < len(group_bins) - 2
        else f"свыше {group_bins[i]:,.0f}"
        for i in range(len(group_bins) - 1)
    ]

    for i in range(len(group_bins) - 1):
        mask = (y_true_values >= group_bins[i]) & (y_true_values < group_bins[i + 1])
        if mask.sum() > 0:
            group_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            results[group_labels[i]] = {
                "mae": group_mae,
                "count": mask.sum(),
                "percentage": mask.sum() / len(y_true) * 100,
            }

    return results


# 1. ПРЕДОБРАБОТКА ДАННЫХ - С СОХРАНЕНИЕМ ВЕСОВ


def preprocess_data(df_train, df_test, target_col="target", id_cols=["id", "dt"]):
    """Комплексная предобработка данных для train и test"""

    print("Начало предобработки...")

    # Копируем данные
    train_processed = df_train.copy()
    test_processed = df_test.copy()

    # Сохраняем ID для submission
    test_ids = test_processed["id"] if "id" in test_processed.columns else None

    # Сохраняем веса если есть
    train_weights = None
    if "w" in train_processed.columns:
        train_weights = train_processed["w"]
        train_processed = train_processed.drop(columns=["w"])
        print("Обнаружены веса (w) в тренировочных данных")

    # Удаление служебных колонок
    cols_to_drop = [col for col in id_cols if col in train_processed.columns]
    if cols_to_drop:
        train_processed = train_processed.drop(columns=cols_to_drop)
        test_processed = test_processed.drop(columns=cols_to_drop)

    # Разделение на признаки и целевую переменную для train
    if target_col in train_processed.columns:
        y_train = train_processed[target_col]
        X_train = train_processed.drop(columns=[target_col])
    else:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена в train данных")

    # Для test данных целевой переменной нет
    X_test = test_processed

    print(
        f"После удаления служебных колонок: X_train {X_train.shape}, X_test {X_test.shape}"
    )

    # ВЫРАВНИВАЕМ СТОЛБЦЫ: оставляем только общие признаки
    common_columns = list(set(X_train.columns) & set(X_test.columns))
    print(f"Общих признаков: {len(common_columns)}")

    X_train = X_train[common_columns]
    X_test = X_test[common_columns]

    # Удаление признаков с >70% пропусков на основе train данных
    missing_percent = X_train.isnull().sum() / len(X_train) * 100
    high_missing_cols = missing_percent[missing_percent > 70].index
    print(f"Удаление признаков с >70% пропусков: {len(high_missing_cols)} признаков")

    if len(high_missing_cols) > 0:
        X_train = X_train.drop(columns=high_missing_cols)
        X_test = X_test.drop(
            columns=[col for col in high_missing_cols if col in X_test.columns]
        )

    print(
        f"После удаления признаков с пропусками: X_train {X_train.shape}, X_test {X_test.shape}"
    )

    # Разделение на числовые и категориальные
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(f"Числовых признаков: {len(numeric_features)}")
    print(f"Категориальных признаков: {len(categorical_features)}")

    # Обработка категориальных признаков
    if categorical_features:
        print("Кодирование категориальных признаков...")
        for col in categorical_features:
            if col in X_train.columns:
                # Заполняем пропуски перед кодированием
                X_train[col] = X_train[col].fillna("missing")
                if col in X_test.columns:
                    X_test[col] = X_test[col].fillna("missing")

                # Объединяем все возможные значения
                all_categories = pd.concat([X_train[col], X_test[col]], axis=0).unique()  # type: ignore

                le = LabelEncoder()
                le.fit(all_categories)

                X_train[col] = le.transform(X_train[col])
                X_test[col] = le.transform(X_test[col])

    # Импутация пропусков для числовых признаков
    if numeric_features:
        print("Импутация пропусков для числовых признаков...")
        imputer = SimpleImputer(strategy="median")

        # Применяем импутацию
        X_train_numeric = X_train[numeric_features]
        X_test_numeric = X_test[numeric_features]

        X_train[numeric_features] = imputer.fit_transform(X_train_numeric)
        X_test[numeric_features] = imputer.transform(X_test_numeric)

    # Обработка выбросов (винзоризация на уровне 1-99 перцентилей) только для train
    print("Обработка выбросов (винзоризация)...")
    for col in numeric_features:
        if col in X_train.columns:
            lower = X_train[col].quantile(0.01)
            upper = X_train[col].quantile(0.99)
            X_train[col] = X_train[col].clip(lower=lower, upper=upper)
            # Для test также применяем те же границы
            if col in X_test.columns:
                X_test[col] = X_test[col].clip(lower=lower, upper=upper)

    print(f"Финальные размеры: X_train {X_train.shape}, X_test {X_test.shape}")

    return X_train, X_test, y_train, test_ids, train_weights


# 2. FEATURE ENGINEERING


def create_features(X_train, X_test):
    """Создание дополнительных признаков для train и test"""

    X_train_fe = X_train.copy()
    X_test_fe = X_test.copy()

    print("Создание новых признаков...")

    # Агрегированные признаки по группам

    # 1. Соотношение кредитовых и дебетовых оборотов
    if (
        "turn_cur_cr_avg_v2" in X_train_fe.columns
        and "turn_cur_db_avg_v2" in X_train_fe.columns
    ):
        X_train_fe["credit_debit_ratio"] = X_train_fe["turn_cur_cr_avg_v2"] / (
            X_train_fe["turn_cur_db_avg_v2"] + 1
        )
        if (
            "turn_cur_cr_avg_v2" in X_test_fe.columns
            and "turn_cur_db_avg_v2" in X_test_fe.columns
        ):
            X_test_fe["credit_debit_ratio"] = X_test_fe["turn_cur_cr_avg_v2"] / (
                X_test_fe["turn_cur_db_avg_v2"] + 1
            )

    # 2. Доля остатков от оборотов
    if (
        "curr_rur_amt_cm_avg" in X_train_fe.columns
        and "turn_cur_db_avg_v2" in X_train_fe.columns
    ):
        X_train_fe["balance_to_turnover"] = X_train_fe["curr_rur_amt_cm_avg"] / (
            X_train_fe["turn_cur_db_avg_v2"] + 1
        )
        if (
            "curr_rur_amt_cm_avg" in X_test_fe.columns
            and "turn_cur_db_avg_v2" in X_test_fe.columns
        ):
            X_test_fe["balance_to_turnover"] = X_test_fe["curr_rur_amt_cm_avg"] / (
                X_test_fe["turn_cur_db_avg_v2"] + 1
            )

    # 3. Интенсивность использования кредитных продуктов
    if "hdb_bki_total_products" in X_train_fe.columns and "age" in X_train_fe.columns:
        X_train_fe["products_per_age"] = X_train_fe["hdb_bki_total_products"] / (
            X_train_fe["age"] + 1
        )
        if "hdb_bki_total_products" in X_test_fe.columns and "age" in X_test_fe.columns:
            X_test_fe["products_per_age"] = X_test_fe["hdb_bki_total_products"] / (
                X_test_fe["age"] + 1
            )

    # 4. Долговая нагрузка
    if (
        "hdb_outstand_sum" in X_train_fe.columns
        and "salary_6to12m_avg" in X_train_fe.columns
    ):
        X_train_fe["debt_to_income"] = X_train_fe["hdb_outstand_sum"] / (
            X_train_fe["salary_6to12m_avg"] + 1
        )
        if (
            "hdb_outstand_sum" in X_test_fe.columns
            and "salary_6to12m_avg" in X_test_fe.columns
        ):
            X_test_fe["debt_to_income"] = X_test_fe["hdb_outstand_sum"] / (
                X_test_fe["salary_6to12m_avg"] + 1
            )

    # 5. Средний оборот по транзакции
    transaction_cols_train = [
        col
        for col in X_train_fe.columns
        if "by_category" in col or "transaction" in col
    ]
    if transaction_cols_train:
        X_train_fe["avg_transaction_amount"] = X_train_fe[transaction_cols_train].mean(
            axis=1
        )

    transaction_cols_test = [
        col for col in X_test_fe.columns if "by_category" in col or "transaction" in col
    ]
    if transaction_cols_test:
        X_test_fe["avg_transaction_amount"] = X_test_fe[transaction_cols_test].mean(
            axis=1
        )

    new_features_count = len(X_train_fe.columns) - len(X_train.columns)
    print(f"Создано {new_features_count} новых признаков")

    # Снова выравниваем столбцы после создания новых признаков
    common_columns_after_fe = list(set(X_train_fe.columns) & set(X_test_fe.columns))
    X_train_fe = X_train_fe[common_columns_after_fe]
    X_test_fe = X_test_fe[common_columns_after_fe]

    return X_train_fe, X_test_fe


# 3. ОТБОР ПРИЗНАКОВ


def select_features(X_train, X_test, y_train, n_features=100):
    """Отбор наиболее важных признаков на train и применение к test"""

    print(f"Отбор топ-{n_features} признаков с помощью Random Forest...")

    # Обучение Random Forest для получения важности признаков
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Важность признаков
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Выбор топ-N признаков
    top_features = feature_importance.head(n_features)["feature"].tolist()

    print("Топ-20 наиболее важных признаков:")
    print(feature_importance.head(20))

    # Применяем отбор к train и test
    X_train_selected = X_train[top_features]

    # Для test берем только те признаки, которые есть в обоих наборах
    available_test_features = [feat for feat in top_features if feat in X_test.columns]
    X_test_selected = X_test[available_test_features]

    print(
        f"После отбора признаков: X_train {X_train_selected.shape}, X_test {X_test_selected.shape}"
    )

    return X_train_selected, X_test_selected, feature_importance


# 4. ОБУЧЕНИЕ МОДЕЛЕЙ С WMAE


def train_models(X_train, X_test, y_train, weights_train=None):
    """Обучение нескольких моделей и сравнение результатов с WMAE"""

    models = {}
    predictions = {}
    metrics = []

    # Разделение train на train/validation
    if weights_train is not None:
        X_tr, X_val, y_tr, y_val, weights_tr, weights_val = train_test_split(
            X_train, y_train, weights_train, test_size=0.2, random_state=42
        )
    else:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        weights_val = None

    print(f"Разделение данных: train {X_tr.shape}, validation {X_val.shape}")

    # Масштабирование
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    # 1. XGBoost
    print("\n" + "=" * 80)
    print("1. Обучение XGBoost...")
    print("=" * 80)

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    xgb_model.fit(X_tr_scaled, y_tr, eval_set=[(X_val_scaled, y_val)], verbose=100)

    models["XGBoost"] = xgb_model
    predictions["XGBoost"] = xgb_model.predict(X_val_scaled)

    # 2. LightGBM
    print("\n" + "=" * 80)
    print("2. Обучение LightGBM...")
    print("=" * 80)

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        verbose=-1,
    )

    lgb_model.fit(
        X_tr_scaled, y_tr, eval_set=[(X_val_scaled, y_val)], eval_metric="mae"
    )

    models["LightGBM"] = lgb_model
    predictions["LightGBM"] = lgb_model.predict(X_val_scaled)

    # 3. CatBoost
    print("\n" + "=" * 80)
    print("3. Обучение CatBoost...")
    print("=" * 80)

    cat_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        early_stopping_rounds=50,
        verbose=100,
    )

    cat_model.fit(X_tr_scaled, y_tr, eval_set=(X_val_scaled, y_val))

    models["CatBoost"] = cat_model
    predictions["CatBoost"] = cat_model.predict(X_val_scaled)

    # 4. Random Forest
    print("\n" + "=" * 80)
    print("4. Обучение Random Forest...")
    print("=" * 80)

    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1, verbose=1
    )

    rf_model.fit(X_tr_scaled, y_tr)

    models["Random Forest"] = rf_model
    predictions["Random Forest"] = rf_model.predict(X_val_scaled)

    # Оценка качества всех моделей с WMAE
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ (С WMAE)")
    print("=" * 80)

    for model_name, y_pred in predictions.items():
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100

        # WMAE
        if weights_val is not None:
            wmae = weighted_mae(y_val, y_pred, weights_val)
        else:
            wmae = mae  # Если весов нет, WMAE = MAE

        # Анализ по группам доходов
        income_groups = calculate_wmae_by_income_group(y_val, y_pred, y_val)

        metrics.append(
            {
                "Model": model_name,
                "MAE": mae,
                "WMAE": wmae,
                "RMSE": rmse,
                "R²": r2,
                "MAPE": mape,
                "Income_Groups": income_groups,
            }
        )

        print(f"\n{model_name}:")
        print(f"  MAE:  {mae:,.2f}")
        print(
            f"  WMAE: {wmae:,.2f}"
            + (" (взвешенная)" if weights_val is not None else "")
        )
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Вывод MAE по группам доходов
        print("  MAE по группам доходов:")
        for group, stats in income_groups.items():
            print(
                f"    {group}: {stats['mae']:,.2f} ({stats['count']} samples, {stats['percentage']:.1f}%)"
            )

    metrics_df = pd.DataFrame(metrics).sort_values("WMAE")

    return models, predictions, metrics_df, scaler, X_val, y_val, weights_val


# 5. АНСАМБЛИРОВАНИЕ И ПРЕДСКАЗАНИЯ НА TEST С WMAE


def create_ensemble_and_predict(
    models, X_val_scaled, X_test_scaled, y_val, weights_val=None
):
    """Создание ансамбля моделей и предсказание на test с WMAE"""

    print("\n" + "=" * 80)
    print("СОЗДАНИЕ АНСАМБЛЯ И ПРЕДСКАЗАНИЯ (С WMAE)")
    print("=" * 80)

    # Предсказания на validation
    val_predictions = {}
    test_predictions = {}

    for model_name, model in models.items():
        val_predictions[model_name] = model.predict(X_val_scaled)
        test_predictions[model_name] = model.predict(X_test_scaled)

    # Ансамбли на validation
    # Простое усреднение
    y_pred_ensemble_mean = np.mean([pred for pred in val_predictions.values()], axis=0)

    # Медиана
    y_pred_ensemble_median = np.median(
        [pred for pred in val_predictions.values()], axis=0
    )

    # Взвешенное усреднение (веса пропорциональны обратной MAE)
    weights = []
    for pred in val_predictions.values():
        mae = mean_absolute_error(y_val, pred)
        weights.append(1 / mae)

    weights = np.array(weights) / np.sum(weights)
    y_pred_ensemble_weighted = np.average(
        [pred for pred in val_predictions.values()], axis=0, weights=weights
    )

    # Оценка ансамблей на validation с WMAE
    ensembles_val = {
        "Ensemble (Mean)": y_pred_ensemble_mean,
        "Ensemble (Median)": y_pred_ensemble_median,
        "Ensemble (Weighted)": y_pred_ensemble_weighted,
    }

    ensemble_metrics = []
    for name, y_pred in ensembles_val.items():
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100

        # WMAE для ансамблей
        if weights_val is not None:
            wmae = weighted_mae(y_val, y_pred, weights_val)
        else:
            wmae = mae

        # Анализ по группам доходов для ансамблей
        income_groups = calculate_wmae_by_income_group(y_val, y_pred, y_val)

        ensemble_metrics.append(
            {
                "Ensemble": name,
                "MAE": mae,
                "WMAE": wmae,
                "RMSE": rmse,
                "R²": r2,
                "MAPE": mape,
                "Income_Groups": income_groups,
            }
        )

        print(f"\n{name} (validation):")
        print(f"  MAE:  {mae:,.2f}")
        print(
            f"  WMAE: {wmae:,.2f}"
            + (" (взвешенная)" if weights_val is not None else "")
        )
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        # Вывод MAE по группам доходов для ансамблей
        print("  MAE по группам доходов:")
        for group, stats in income_groups.items():
            print(
                f"    {group}: {stats['mae']:,.2f} ({stats['count']} samples, {stats['percentage']:.1f}%)"
            )

    # Создание ансамблей для test данных
    ensembles_test = {
        "Ensemble_Mean": np.mean([pred for pred in test_predictions.values()], axis=0),
        "Ensemble_Median": np.median(
            [pred for pred in test_predictions.values()], axis=0
        ),
        "Ensemble_Weighted": np.average(
            [pred for pred in test_predictions.values()], axis=0, weights=weights
        ),
    }

    return (
        ensembles_test,
        ensemble_metrics,
        pd.DataFrame(ensemble_metrics),
        test_predictions,
        weights,
    )


# 6. СОХРАНЕНИЕ И ЗАГРУЗКА МОДЕЛЕЙ


def save_models_and_artifacts(
    models,
    scaler,
    feature_importance,
    ensemble_weights,
    metrics_df,
    ensemble_metrics_df,
    submission,
    model_dir="saved_models",
):
    """Сохранение моделей и всех артефактов"""

    print(f"\nСОХРАНЕНИЕ МОДЕЛЕЙ И АРТЕФАКТОВ В ПАПКУ '{model_dir}'...")

    # Создаем папку если не существует
    os.makedirs(model_dir, exist_ok=True)

    # Текущая дата для версионирования
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохраняем модели
    for model_name, model in models.items():
        model_filename = os.path.join(model_dir, f"{model_name}_{current_time}.pkl")

        if model_name == "XGBoost":
            # Для XGBoost используем встроенный метод
            model.save_model(
                os.path.join(model_dir, f"{model_name}_{current_time}.json")
            )
        elif model_name == "LightGBM":
            # Для LightGBM используем встроенный метод
            model.booster_.save_model(
                os.path.join(model_dir, f"{model_name}_{current_time}.txt")
            )
        elif model_name == "CatBoost":
            # Для CatBoost используем встроенный метод
            model.save_model(
                os.path.join(model_dir, f"{model_name}_{current_time}.cbm")
            )

        # Дополнительно сохраняем все модели в pickle
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        print(f"Сохранена модель: {model_filename}")

    # Сохраняем scaler
    scaler_filename = os.path.join(model_dir, f"scaler_{current_time}.pkl")
    with open(scaler_filename, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Сохранен scaler: {scaler_filename}")

    # Сохраняем веса ансамбля
    weights_filename = os.path.join(model_dir, f"ensemble_weights_{current_time}.pkl")
    with open(weights_filename, "wb") as f:
        pickle.dump(ensemble_weights, f)
    print(f"Сохранены веса ансамбля: {weights_filename}")

    # Сохраняем feature importance
    feature_importance_filename = os.path.join(
        model_dir, f"feature_importance_{current_time}.csv"
    )
    feature_importance.to_csv(feature_importance_filename, index=False)
    print(f"Сохранена важность признаков: {feature_importance_filename}")

    # Сохраняем метрики
    metrics_filename = os.path.join(model_dir, f"model_metrics_{current_time}.csv")
    metrics_df.to_csv(metrics_filename, index=False)

    ensemble_metrics_filename = os.path.join(
        model_dir, f"ensemble_metrics_{current_time}.csv"
    )
    ensemble_metrics_df.to_csv(ensemble_metrics_filename, index=False)
    print(f"Сохранены метрики: {metrics_filename}, {ensemble_metrics_filename}")

    # Сохраняем submission
    submission_filename = os.path.join(model_dir, f"submission_{current_time}.csv")
    submission.to_csv(submission_filename, index=False)
    print(f"Сохранен submission файл: {submission_filename}")

    # Создаем файл с информацией о версии
    info_content = f"""
МОДЕЛИ ДЛЯ ПРОГНОЗИРОВАНИЯ ДОХОДА
Дата создания: {current_time}
Количество моделей: {len(models)}
Лучшая модель: {metrics_df.iloc[0]["Model"]}
Лучший WMAE: {metrics_df.iloc[0]["WMAE"]:,.2f}
Лучший MAE: {metrics_df.iloc[0]["MAE"]:,.2f}
Лучший R²: {metrics_df.iloc[0]["R²"]:.4f}

Список моделей:
{chr(10).join([f"- {name}" for name in models.keys()])}
"""

    info_filename = os.path.join(model_dir, f"model_info_{current_time}.txt")
    with open(info_filename, "w", encoding="utf-8") as f:
        f.write(info_content)
    print(f"Сохранена информация о моделях: {info_filename}")

    return current_time


# 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ


def visualize_results(y_val, predictions, feature_importance, models, metrics_df):
    """Визуализация результатов"""

    print("Визуализация результатов...")

    # 1. Сравнение предсказаний с реальными значениями
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        ax = axes[idx // 2, idx % 2]
        ax.scatter(y_val, y_pred, alpha=0.5)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--", lw=2)
        ax.set_xlabel("Реальные значения")
        ax.set_ylabel("Предсказанные значения")
        ax.set_title(f"{model_name}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions_comparison.png", dpi=300, bbox_inches="tight")
    print("График сравнения предсказаний сохранен: predictions_comparison.png")

    # 2. Feature Importance (топ-20)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = feature_importance.head(20)
    ax.barh(range(len(top_20)), top_20["importance"])
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20["feature"])
    ax.set_xlabel("Важность признака")
    ax.set_title("Топ-20 наиболее важных признаков")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    print("График важности признаков сохранен: feature_importance.png")

    # 3. Сравнение метрик моделей
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # MAE
    axes[0, 0].bar(metrics_df["Model"], metrics_df["MAE"], color="skyblue")
    axes[0, 0].set_title("MAE по моделям")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # WMAE
    axes[0, 1].bar(metrics_df["Model"], metrics_df["WMAE"], color="lightcoral")
    axes[0, 1].set_title("WMAE по моделям")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # R²
    axes[1, 0].bar(metrics_df["Model"], metrics_df["R²"], color="lightgreen")
    axes[1, 0].set_title("R² по моделям")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # MAPE
    axes[1, 1].bar(metrics_df["Model"], metrics_df["MAPE"], color="gold")
    axes[1, 1].set_title("MAPE по моделям")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("model_metrics_comparison.png", dpi=300, bbox_inches="tight")
    print("График сравнения метрик моделей сохранен: model_metrics_comparison.png")


# 8. ГЛАВНАЯ ФУНКЦИЯ


def main():
    """Главная функция для запуска всего pipeline"""

    print("=" * 80)
    print("ПРОГНОЗИРОВАНИЕ ДОХОДА КЛИЕНТА (С WMAE)")
    print("=" * 80)

    # 1. Preprocessing
    print("\n1. ПРЕДОБРАБОТКА ДАННЫХ...")
    X_train, X_test, y_train, test_ids, train_weights = preprocess_data(
        df_train, df_test
    )

    # 2. Feature Engineering
    print("\n2. FEATURE ENGINEERING...")
    X_train_fe, X_test_fe = create_features(X_train, X_test)

    # 3. Feature Selection
    print("\n3. ОТБОР ПРИЗНАКОВ...")
    X_train_selected, X_test_selected, feature_importance = select_features(
        X_train_fe, X_test_fe, y_train, n_features=100
    )

    # 4. Обучение моделей
    print("\n4. ОБУЧЕНИЕ МОДЕЛЕЙ...")
    models, predictions, metrics_df, scaler, X_val, y_val, weights_val = train_models(
        X_train_selected, X_test_selected, y_train, train_weights
    )

    # 5. Ансамблирование и предсказания на test
    print("\n5. АНСАМБЛИРОВАНИЕ И ПРЕДСКАЗАНИЯ...")

    # Масштабирование test данных
    X_test_scaled = scaler.transform(X_test_selected)

    # Получаем масштабированные validation данные
    X_val_scaled = scaler.transform(X_val)

    (
        ensembles_test,
        ensemble_metrics,
        ensemble_metrics_df,
        test_predictions,
        ensemble_weights,
    ) = create_ensemble_and_predict(
        models, X_val_scaled, X_test_scaled, y_val, weights_val
    )

    # 6. Сохранение предсказаний
    print("\n6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ...")

    # Сохраняем предсказания ансамбля (взвешенного) на test
    test_predictions_final = ensembles_test["Ensemble_Weighted"]

    # Создаем submission файл
    submission = pd.DataFrame(
        {"id": test_ids, "predicted_income": test_predictions_final}
    )

    submission.to_csv("income_predictions.csv", index=False)
    print("Предсказания сохранены в файл: income_predictions.csv")

    # Сохраняем метрики
    metrics_df.to_csv("model_metrics.csv", index=False)
    ensemble_metrics_df.to_csv("ensemble_metrics.csv", index=False)
    feature_importance.to_csv("feature_importance.csv", index=False)

    # 7. Сохранение моделей и артефактов
    print("\n7. СОХРАНЕНИЕ МОДЕЛЕЙ И АРТЕФАКТОВ...")
    saved_version = save_models_and_artifacts(
        models,
        scaler,
        feature_importance,
        ensemble_weights,
        metrics_df,
        ensemble_metrics_df,
        submission,
    )

    # 8. Визуализация
    print("\n8. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ...")
    visualize_results(y_val, predictions, feature_importance, models, metrics_df)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)
    print("Файл с предсказаниями: income_predictions.csv")
    print(f"Версия моделей: {saved_version}")
    print(f"Размер файла с предсказаниями: {submission.shape}")

    # Вывод итоговых результатов
    print("\nИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("Лучшая модель по WMAE:", metrics_df.iloc[0]["Model"])
    print(f"Лучший WMAE: {metrics_df.iloc[0]['WMAE']:,.2f}")
    print(f"Лучший MAE: {metrics_df.iloc[0]['MAE']:,.2f}")
    print(f"Лучший R²: {metrics_df.iloc[0]['R²']:.4f}")

    return models, scaler, feature_importance, submission, metrics_df, saved_version


# Запуск главной функции
if __name__ == "__main__":
    models, scaler, feature_importance, submission, metrics_df, saved_version = main()
