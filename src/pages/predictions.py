import pandas as pd
import streamlit as st

from src.core.inference import inference
from src.utils import BankClientAnalyzer


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
            nrows=2000,
        )
        # st.write(new_data)

        if st.button("Посчитать доходы", icon=":material/psychology:"):
            with st.spinner("Гоблины думают..."):
                try:
                    # Вызываем функцию предсказания
                    df_results, _ = inference(new_data)

                    # Отображаем результаты
                    st.subheader("Результаты предсказания")
                    st.markdown(
                        f"Наученные думать гоблинские модели обработали {len(df_results)} клиентов!"
                    )

                    st.subheader("Что мы можем сказать, основываясь на этих данных?")

                    analyzer = BankClientAnalyzer(df_results)

                    st.markdown(analyzer.get_main_stats())

                    risk_clients = analyzer.get_clients_with_risks()
                    st.markdown("### 🚩 КЛИЕНТЫ, ТРЕБУЮЩИЕ ВНИМАНИЯ")
                    st.markdown(
                        "Оценки наших моделей при работе с данными клиентов сильно расходятся, что выражается в значении коэффициента вариации (CV)."
                    )
                    st.markdown(
                        "Это связано как с противоречиями в данных, так и в их отсутствии."
                    )
                    st.dataframe(risk_clients)

                    vip_clients = analyzer.get_vip_clients()
                    st.markdown("### 💎 VIP-КЛИЕНТЫ")
                    st.markdown(
                        "Люди с относительно большими доходами, готовые взять на себя долговые обязательства."
                    )

                    st.dataframe(vip_clients)

                    st.header(
                        "А здесь мы показываем прогнозы моделей в ансамбле и по отдельности"
                    )
                    st.markdown("Это техническая информация")

                    df_results = df_results.rename(
                        columns={"prediction": "Доход, руб", "id": "Id пользователя"}
                    )
                    st.dataframe(df_results)

                except Exception as e:
                    st.error(f"Произошла ошибка: {e}")


predictions = st.Page(
    page=page,
    title="5 Предсказание",
    icon=":material/manage_search:",
    url_path="predictions",
)
