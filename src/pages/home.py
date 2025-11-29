import streamlit as st


def page():
    CARD_TEMPLATE = """
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    st.title(":material/monitoring: Income Genie")
    st.subheader("Виртуальный ассистент, предсказывающий доходы ваших клиентов")
    st.write("""
    Прибыль банка во многом зависит от того, насколько точно он может оценить финансовые возможности 
    клиентов. Ошибки в оценке доходов могут привести либо к рискам при выдаче кредитов, либо к ухудше
    нию пользовательского опыта, когда люди получают нерелевантные или менее выгодные предложения. 
    Прогнозирование дохода помогает точнее оценивать платежеспособность клиентов, персонализиро
    вать предложения и устанавливать оптимальные условия, одновременно соблюдая требования 
    Центрального банка по контролю кредитоспособности.
    """)
    st.markdown("---")
    st.subheader("Над программой работали")
    st.info("""
    # Поволжские гоблины
    """, icon=":material/star_shine:")

    column1, column2, column3 = st.columns(3)

    with column1:
        st.markdown(
            CARD_TEMPLATE.format(title="Backend", value="Глазунов Тимур"),
            unsafe_allow_html=True,
        )
    with column2:
        st.markdown(
            CARD_TEMPLATE.format(title="ML/DS", value="Мокин Дмитрий"),
            unsafe_allow_html=True,
        )
    with column3:
        st.markdown(
            CARD_TEMPLATE.format(title="ML/DS", value="Елсуков Сергей"),
            unsafe_allow_html=True,
        )


home = st.Page(
    page=page, title="1 Главная страница", icon=":material/home:", url_path="home"
)
