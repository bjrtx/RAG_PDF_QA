import streamlit as st
from utils import (
    prepare_data_for_mistral,
    get_answer)


st.title("RAG application on PDF question answering")

documents, nodes, collection, model, client = prepare_data_for_mistral(
    use_dir=True
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

question_input = st.chat_input("Ask a question:", key="question_input")

if question_input:
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
