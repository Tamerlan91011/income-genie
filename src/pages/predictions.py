import streamlit as st


def page():
    st.title(":material/search_insights: Как это работает?")
    st.subheader("Сейчас покажем!")
    st.markdown("---")

    st.info(f"""
    **95% Confidence Interval:**
    - Lower Bound: ${0:,.0f}
    - Upper Bound: ${0:,.0f}
    - Range: ${0 - 0:,.0f}
    """)


predictions = st.Page(
    page=page,
    title="4 Предсказание",
    icon=":material/manage_search:",
    url_path="predictions",
)
