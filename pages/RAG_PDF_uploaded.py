import streamlit as st

from src.utils import upload_files, prepare_data_for_mistral, get_answer, display_file

st.set_page_config(layout="wide")
st.title("Document question answering")

col1, col2 = st.columns(2)

with col2:
    uploaded_files = upload_files()

for uploaded_file in uploaded_files:
    with col1:
        documents, nodes, collection, model, client = prepare_data_for_mistral(
            uploaded_file=uploaded_file
        )

        section_key = "RAG_PDF"
        if f"messages_{section_key}" not in st.session_state:
            st.session_state[f"messages_{section_key}"] = []

        question_input = st.text_input(
            "Ask a question:", key=f"question_input_{section_key}"
        )

        if st.button("Ask"):
            with st.spinner('Awaiting model answer...'):
                try:
                    if collection is not None:
                        chat_response = get_answer(
                            question_input, collection, model, client, prompt_key="RAG_PDF"
                        )
                        st.session_state[f"messages_{section_key}"].append(
                            {"role": "user", "content": question_input}
                        )
                        st.session_state[f"messages_{section_key}"].append(
                            {"role": "assistant", "content": chat_response}
                        )
                except Exception as e:
                    st.session_state[f"messages_{section_key}"].append(
                        {"role": "assistant", "content": f"Error: {str(e)}"}
                    )
                    st.error("Failed to process the question.", icon='🚨')

        for message in st.session_state[f"messages_{section_key}"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    with col2:
        display_file(uploaded_file)
