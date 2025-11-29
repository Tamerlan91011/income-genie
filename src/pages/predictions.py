import streamlit as st


def page():
    st.title(":material/search_insights: Как это работает?")
    st.subheader("Сейчас покажем!")
    st.markdown("---")

    st.info(f"""
    **95% Интервалы уверенности:**
    - Нижний предел: ${0:,.0f}
    - Верхний предел: ${0:,.0f}
    - Разброс: ${0 - 0:,.0f}
    """)


predictions = st.Page(
    page=page,
    title="4 Предсказание",
    icon=":material/manage_search:",
    url_path="predictions",
)
