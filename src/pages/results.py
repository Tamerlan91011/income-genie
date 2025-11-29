import streamlit as st


def page():
    st.title(":material/checklist: Результаты прогнозов")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", None)
    with col2:
        st.metric("MAE", None)
    with col3:
        st.metric("RMSE", None)
    with col4:
        st.metric("R² Score", None)


results = st.Page(page=page, title="5 Результаты", icon=":material/lab_profile:", url_path="results")
