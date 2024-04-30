import streamlit as st

from src.utils import upload_files, prepare_data_for_mistral, get_summary, display_file

st.set_page_config(layout="wide")
st.title("Document summarizer")

if 'summaries' not in st.session_state:
    st.session_state.summaries = {}

col1, col2 = st.columns(2)

with col1:
    if st.button('Erase all saved summaries'):
        st.session_state.summaries.clear()

with col2:
    uploaded_files = upload_files()

for i, file in enumerate(uploaded_files):
    with col2:
        st.divider()
        display_file(file)
    with col1:
        st.divider()
        summary = st.session_state.summaries.get(file.name)
        if summary is None:
            with st.spinner(f"Preparing a summary for {file.name}..."):
                documents, nodes, collection, model, client = prepare_data_for_mistral(
                    uploaded_file=file, include_collection=False
                )
                summary = get_summary(documents, client, model)
            st.session_state.summaries[file.name] = summary
        st.markdown(summary)
