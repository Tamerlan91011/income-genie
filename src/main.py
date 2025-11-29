import streamlit as st

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.pages.home import home
from src.pages.data_analysis import data_analysis
from src.pages.model_training import model_training
from src.pages.predictions import predictions
from src.pages.results import results

# "1 Главная страница"
# "2 Анализ данных"
# "3 Обучение модели"
# "4 Предсказание"
# "5 Результаты"

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .metric-title {
            font-size: 14px;
            color: #666;
            font-weight: 600;
        }
        .metric-value {
            font-size: 28px;
            color: #1f77b4;
            font-weight: bold;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pg = st.navigation([home, data_analysis, model_training, predictions, results])
    pg.run()
