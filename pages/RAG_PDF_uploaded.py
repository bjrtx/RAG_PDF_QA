import streamlit as st
from utils import (
    upload_pdf,
    display_pdf,
    prepare_data_for_mistral,
    get_answer)


st.set_page_config(layout="wide")
st.title("PDF question answering")

col1, col2 = st.columns(2)

with col2:
    uploaded_file = upload_pdf()

if uploaded_file is not None:
    with col1:
        documents, nodes, collection, model, client = prepare_data_for_mistral(
            uploaded_file=uploaded_file
            )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        question_input = st.text_input("Ask a question:", key="question_input")

        if st.button("Ask"):
            try:
                if collection is not None:
                    chat_response = get_answer(
                        question_input,
                        collection,
                        model,
                        client
                        )
                    st.session_state.messages.append(
                        {"role": "user", "content": question_input})
                    st.session_state.messages.append(
                        {"role": "assistant", "content": chat_response})
            except Exception as e:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {str(e)}"})
                st.error("Failed to process the question.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    with col2:
        display_pdf(uploaded_file)
