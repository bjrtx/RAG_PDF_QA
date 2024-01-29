from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from dotenv import load_dotenv
from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader
import chromadb
import streamlit as st
import os


def Mistral_llm():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    model = "mistral-tiny"
    client = MistralClient(api_key=api_key)
    return model, client


def load_data():
    documents = SimpleDirectoryReader(
        input_dir="./data",
    ).load_data()

    node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
    base_nodes = node_parser.get_nodes_from_documents(documents)
    return base_nodes


def vectorDB():
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="test", metadata={"hnsw:space": "cosine"})
    base_nodes = load_data()
    for i in range(len(base_nodes)):
        collection.add(
            documents=[base_nodes[i].get_text()],
            metadatas=[{'source ': base_nodes[i].metadata.get('file_name')}],
            ids=[f'{i}']
        )
    return collection


def get_answer(question_input):
    collection = vectorDB()
    dbresults = collection.query(
                query_texts=question_input
            )

    messages = [
        ChatMessage(
            role="user",
            content=f"{question_input}:{dbresults.get('documents')[0][0]} ?"
            )
    ]

    model, client = Mistral_llm()
    chat_response = client.chat(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content


def main():
    st.title("Streamlit Question Answering App")
    st.markdown("##### How can I help you ?")

    # Initialize the chat message history
    if "messages" not in st.session_state.keys():
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question_input = st.chat_input()

    if question_input:
        st.session_state.messages.append(
            {"role": "user", "content": question_input})
        with st.chat_message("user"):
            st.write(question_input)
        chat_response = get_answer(question_input)
        st.session_state.messages.append(
            {"role": "assistant", "content": chat_response})
        with st.chat_message("assistant"):
            st.write(chat_response)


if __name__ == "__main__":
    main()
