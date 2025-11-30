import pandas as pd
import numpy as np


class BankClientAnalyzer:
    """
    Анализатор предсказаний доходов клиентов банка.
    Генерирует бизнес-рекомендации на основе консенсуса моделей.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Инициализация анализатора.

        Args:
            df: Датафрейм с данными предсказаний
        """
        self.df = df
        self.models = [
            "CatBoost_pred",
            "LightGBM_pred",
            "Random Forest_pred",
            "XGBoost_pred",
        ]
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Рассчитывает метрики для анализа."""
        # Отклонение каждой модели от финального предсказания
        for model in self.models:
            self.df[f"{model}_deviation"] = np.abs(
                self.df[model] - self.df["prediction"]
            )

        # Статистика между моделями
        self.df["model_mean"] = self.df[self.models].mean(axis=1)
        self.df["model_std"] = self.df[self.models].std(axis=1)
        self.df["model_min"] = self.df[self.models].min(axis=1)
        self.df["model_max"] = self.df[self.models].max(axis=1)
        self.df["model_range"] = self.df["model_max"] - self.df["model_min"]

        # Коэффициент вариации (CV) - мера неопределённости в %
        self.df["cv"] = (self.df["model_std"] / self.df["prediction"]) * 100

        # Ранжирование моделей по точности
        self.model_accuracy = {}
        for model in self.models:
            self.model_accuracy[model] = np.abs(
                self.df[model] - self.df["prediction"]
            ).mean()

        self.q75 = self.df["prediction"].quantile(0.75)
        self.q25 = self.df["prediction"].quantile(0.25)

        self.high = self.df[self.df["prediction"] > self.q75]
        self.medium = self.df[
            (self.df["prediction"] > self.q25) & (self.df["prediction"] <= self.q75)
        ]
        self.low = self.df[self.df["prediction"] <= self.q25]

        self.low_risk = len(self.df[self.df["cv"] < 10])
        self.moderate_risk = len(self.df[(self.df["cv"] >= 10) & (self.df["cv"] < 15)])
        self.high_risk = len(self.df[(self.df["cv"] >= 15) & (self.df["cv"] < 25)])
        self.critical_risk = len(self.df[self.df["cv"] >= 25])

    def get_main_stats(self) -> str:
        """Генерирует основные метрики по всему портфелю."""
        summary = []

        # Основные метрики
        summary.append("### 📊 ОБЩАЯ СТАТИСТИКА\n")
        summary.append(f"   Количество клиентов: {len(self.df)}\n")
        summary.append(
            f"   Средний прогноз дохода: ₽{self.df['prediction'].mean():,.2f}\n"
        )
        summary.append(f"   Медиана: ₽{self.df['prediction'].median():,.2f}\n")
        summary.append(
            f"   Диапазон: ₽{self.df['prediction'].min():,.2f} - ₽{self.df['prediction'].max():,.2f}\n"
        )
        summary.append(f"   Общая сумма походов: ₽{self.df['prediction'].sum():,.2f}\n")

        # Сегментация
        summary.append("### 👥 СЕГМЕНТАЦИЯ КЛИЕНТОВ\n")
        q75 = self.df["prediction"].quantile(0.75)
        q25 = self.df["prediction"].quantile(0.25)

        high = self.df[self.df["prediction"] > q75]
        medium = self.df[(self.df["prediction"] > q25) & (self.df["prediction"] <= q75)]
        low = self.df[self.df["prediction"] <= q25]

        summary.append(
            f"   🟢 Высокий доход: {len(high)} клиентов, средний ₽{high['prediction'].mean():,.2f}\n"
        )
        summary.append(
            f"   🟡 Средний доход: {len(medium)} клиентов, средний ₽{medium['prediction'].mean():,.2f}\n"
        )
        summary.append(
            f"   🔵 Низкий доход: {len(low)} клиентов, средний ₽{low['prediction'].mean():,.2f}\n"
        )

        # Распределение по уровню риска
        summary.append("### ⚠️  РАСПРЕДЕЛЕНИЕ ПО УРОВНЮ РИСКА\n")

        summary.append(f"   🟢 Низкий риск (CV < 10%): {self.low_risk} клиентов\n")
        summary.append(f"   🟡 Умеренный (CV 10-15%): {self.moderate_risk} клиентов\n")
        summary.append(f"   🟠 Высокий (CV 15-25%): {self.high_risk} клиентов\n")
        summary.append(
            f"   🔴 Критический (CV >= 25%): {self.critical_risk} клиентов\n"
        )

        return "\n".join(summary)

    def get_clients_with_risks(self) -> tuple[str, dict]:
        """Генерирует сводку по крайним, рискованным случаям."""
        summary = []
        clients: dict = {}

        # Флаг-случаи
        if self.critical_risk > 0:
            summary.append("### 🚩 КЛИЕНТЫ, ТРЕБУЮЩИЕ ВНИМАНИЯ\n")

            critical = self.df[self.df["cv"] >= 25].sort_values("cv", ascending=False)

            for idx, row in critical.iterrows():
                clients[int(row["id"])] = (
                    f"Коэффициент вариации {row['cv']:.2f}%, прогноз ₽{row['prediction']:,.2f}"
                )

            summary.append("")

        return "\n".join(summary), clients

    def get_vip_clients(self):
        clients:  dict[int, str] = {}
        
        # VIP список
        if len(self.high) > 0:
            vip = self.df[self.df["prediction"] > self.q75].sort_values(
                "prediction", ascending=False
            )
            for idx, row in vip.iterrows():
                risk_emoji = "🟢" if row["cv"] < 12 else "🟡"
                
                clients[int(row["id"])] = (
                    f" {risk_emoji} Коэффициент вариации {row['cv']:.2f}%, прогноз ₽{row['prediction']:,.2f}"
                )
        
        return clients
