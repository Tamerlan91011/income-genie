import streamlit as st


def page():
    st.title(":material/science: Процесс обучения модели")
    st.markdown("---")

    st.info(
        "✓ Тут ML эксперты расскажут, как работает процесс прогноза и какие этапы он в себя включает"
    )


model_training = st.Page(
    page,
    title="3 Обучение модели",
    icon=":material/model_training:",
    url_path="model_training",
)
