import streamlit as st
from utils import Embeddings, display_pdf, Mistral_API, summary
from io import BytesIO


def upload_pdf() -> BytesIO:
    uploaded_file = st.file_uploader("Download PDF", type="pdf")
    return uploaded_file


st.title("PDF summary")
col1, col2 = st.columns(spec=[2, 1], gap="large")

with col2:
    uploaded_file = upload_pdf()

if uploaded_file is not None:
    with col1:
        embeddings = Embeddings(uploaded_file)
        nodes = embeddings.parse_PDF()
        model, client = Mistral_API()
        resume = summary(nodes, client, model)
        #st.markdown(resume)
    with col2:
        display_pdf(uploaded_file)
