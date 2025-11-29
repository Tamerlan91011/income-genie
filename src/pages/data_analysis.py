import streamlit as st


def page():
    st.title(":material/article_person: Что мы знаем о вашем клиенте?")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Количество пользователей", 9999)
    with col2:
        st.metric("Количество признаков", 222)
    with col3:
        st.metric("Значений с пропусками", 9999)
    with col4:
        st.metric("Размер файла", "Много KB")


data_analysis = st.Page(
    page=page,
    title="2 Анализ данных",
    icon=":material/bar_chart:",
    url_path="data_analysis",
)
