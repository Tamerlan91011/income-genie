import streamlit as st

def page():
    st.title(":material/monitoring: Income Genie")
    st.subheader("Виртуальный ассистент, предсказывающий доходы ваших клиентов")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-title">Всего записей</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(1),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-title">Использованные признаки</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(2),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
        <div class="metric-card">
            <div class="metric-title">Моделей обучено</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(3),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Анализ признаков")
    st.write("""
            Все, что мы знаем про признаки из их описания
    """)


home = st.Page(
    page=page, title="1 Главная страница", icon=":material/home:", url_path="home"
)
