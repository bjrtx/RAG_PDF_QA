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
    try:
        client = MistralClient(api_key=api_key)
        return model, client
    except Exception as e:
        st.error(f"Failed to initialize Mistral Client: {str(e)}")
        return None, None


def load_data():
    try:
        documents = SimpleDirectoryReader(input_dir="./data").load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
        base_nodes = node_parser.get_nodes_from_documents(documents)
        return base_nodes
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return []


def vectorDB():
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="test",
            metadata={"hnsw:space": "cosine"})
        base_nodes = load_data()
        for i, node in enumerate(base_nodes):
            collection.add(
                documents=[node.get_text()],
                metadatas=[{'source': node.metadata.get('file_name')}],
                ids=[f'{i}'])
        return collection
    except Exception as e:
        st.error(f"Error setting up the vector database: {str(e)}")
        return None


def get_answer(question_input):
    try:
        collection = vectorDB()
        if collection is not None:
            dbresults = collection.query(query_texts=[question_input])
            if dbresults and dbresults.get('documents'):
                content = dbresults.get('documents')[0][0]
                messages = [
                    ChatMessage(
                        role="user",
                        content=f"{question_input}:{content} ?")
                        ]
                model, client = Mistral_llm()
                if client is not None:
                    chat_response = client.chat(model=model, messages=messages)
                    return chat_response.choices[0].message.content
                else:
                    return "Mistral Client is not initialized."
            else:
                return "No results found in the database."
        else:
            return "Vector database is not initialized."
    except Exception as e:
        st.error(f"Failed to get answer: {str(e)}")
        return "An error occurred while processing your question."


def main():
    st.title("RAG Question Answering App")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    question_input = st.chat_input("Ask a question:", key="question_input")

    if question_input:
        try:
            chat_response = get_answer(question_input)
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question_input
                }
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": chat_response
                }
            )
        except Exception as e:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error: {str(e)}"
                }
            )
            st.error("Failed to process the question.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


if __name__ == "__main__":
    main()
