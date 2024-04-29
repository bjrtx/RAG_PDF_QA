import sys

import streamlit as st

from src.utils import upload_file, prepare_data_for_mistral, get_summary, display_file

st.set_page_config(layout="wide")
st.title("Document summary")

col1, col2 = st.columns(2)

with col2:
    uploaded_file = upload_file()

if uploaded_file is not None:
    with col2:
        display_file(uploaded_file)
    with col1:
        with st.spinner("Preparing a summary..."):
            documents, nodes, collection, model, client = prepare_data_for_mistral(
                uploaded_file=uploaded_file, include_collection=False
            )
            resume = get_summary(documents, client, model)
        st.markdown(resume)
