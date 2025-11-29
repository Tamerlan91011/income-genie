import pandas as pd
import streamlit as st

from src.core.inference import inference


def page():
    st.title(":material/search_insights: Как это работает?")

    uploaded_file = st.file_uploader(
        "Загрузите сюда .csv файл, чтобы увидеть **гоблинскую** магию", type="csv"
    )

    if uploaded_file is not None:
        new_data = pd.read_csv(
            uploaded_file,
            decimal=",",
            sep=";",
            engine="python",
            on_bad_lines="warn",
            encoding="UTF-8",
        )
        # st.write(new_data)

        
        if st.button("Сделать предсказание", icon=":material/psychology:"):
            with st.spinner("Гоблины думают..."):
                try:
                    # Вызываем функцию предсказания
                    results, _ = inference(new_data)

                    # Отображаем результаты
                    st.subheader("Результаты предсказания:")
                    st.write(results)

                except Exception as e:
                    st.error(f"Произошла ошибка: {e}")


predictions = st.Page(
    page=page,
    title="4 Предсказание",
    icon=":material/manage_search:",
    url_path="predictions",
)
