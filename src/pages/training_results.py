import pandas as pd
import streamlit as st


def page():
    st.title(":material/checklist: Результаты обучения моделей")
    st.markdown("---")

    df_model_metrics = pd.read_csv("model_metrics.csv")
    df_ensemble_metrics = pd.read_csv("ensemble_metrics.csv")

    df_model_metrics = df_model_metrics.drop("Income_Groups", axis=1)
    df_ensemble_metrics = df_ensemble_metrics.drop("Income_Groups", axis=1)

    st.subheader("Здесь мы приводим статистику наших моделей в их лучшем проявлении")
    st.dataframe(df_model_metrics)

    st.subheader("А здесь мы приводим статистику работы наших моделей в ансамбле")
    st.dataframe(df_ensemble_metrics)


training_results = st.Page(
    page=page, title="4 Результаты обучения", icon=":material/lab_profile:", url_path="results"
)
