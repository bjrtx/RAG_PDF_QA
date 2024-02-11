import streamlit as st
from utils import display_pdf, load_PDF, Mistral_API, summary
from io import BytesIO
from typing import Optional


def upload_pdf() -> Optional[BytesIO]:
    uploaded_file = st.file_uploader("Download PDF", type="pdf")
    if uploaded_file is not None:
        return uploaded_file
    else:
        return None


st.set_page_config(layout="wide")
st.title("PDF summary")
# col1, col2 = st.columns(spec=[2, 1], gap="large")
col1, col2 = st.columns(2)

with col2:
    uploaded_file = upload_pdf()

if uploaded_file is not None:
    with col1:
        docs = load_PDF(uploaded_file)
        model, client = Mistral_API()
        resume = summary(docs, client, model)
        st.markdown(resume)
    with col2:
        display_pdf(uploaded_file)
