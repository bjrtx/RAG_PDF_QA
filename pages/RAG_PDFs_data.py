import streamlit as st

from src.utils import prepare_data_for_mistral, get_answer


st.title("RAG application on PDF question answering")

with st.status("Initializing database..."):
    st.write("Preparing data for Mistral")
    documents, nodes, collection, model, client = prepare_data_for_mistral(use_dir=True)
    st.write(f"Database contains {len(documents)} documents, {len(nodes)} nodes")

section_key = "RAG_PDFs_data"
if f"messages_{section_key}" not in st.session_state:
    st.session_state[f"messages_{section_key}"] = []

question_input = st.chat_input("Ask a question:", key=f"question_input_{section_key}", disabled=not collection)

if question_input:
    with st.spinner("Awaiting model answer..."):
        session_state = st.session_state[f"messages_{section_key}"]
        try:
            chat_response = get_answer(
                question_input,
                collection,
                model,
                client,
                prompt_key="RAG_PDFs_data",
            )
            session_state.append({"role": "user", "content": question_input})
            session_state.append({"role": "assistant", "content": chat_response})
        except Exception as e:
            session_state.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.error("Failed to process the question.", icon="🚨")

for message in st.session_state[f"messages_{section_key}"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
