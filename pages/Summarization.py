import streamlit as st
from utils import (
    upload_pdf,
    display_pdf,
    load_PDF,
    Mistral_API,
    get_summary)


st.set_page_config(layout="wide")
st.title("PDF summary")

col1, col2 = st.columns(2)

with col2:
    uploaded_file = upload_pdf()

if uploaded_file is not None:
    with col1:
        documents = load_PDF(uploaded_file)
        model, client = Mistral_API()
        resume = get_summary(documents, client, model)
        st.markdown(resume)
    with col2:
        display_pdf(uploaded_file)
