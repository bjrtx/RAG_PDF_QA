import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from dotenv import load_dotenv
import os
import chromadb
from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="./data",
).load_data()

node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
base_nodes = node_parser.get_nodes_from_documents(documents)

chroma_client = chromadb.PersistentClient(path="/chroma")
#chroma_client = chromadb.Client()
#chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(name="test", metadata={"hnsw:space": "cosine"})
collection.add(
    #embeddings=[embeddings_batch_response.data[0].embedding, embeddings_batch_response.data[1].embedding],
    documents=[base_nodes[0].get_text(), base_nodes[1].get_text(), base_nodes[2].get_text(), base_nodes[3].get_text()],
    metadatas=[{'source ': base_nodes[0].metadata.get('file_name')}, {'source ': base_nodes[1].metadata.get('file_name')},
                {'source ': base_nodes[2].metadata.get('file_name')}, {'source ': base_nodes[3].metadata.get('file_name')}],
    ids=['0', '1', '2', '3']
)

dbresults = collection.query(
            query_texts="Quels sont les bénifices net de l'entreprise sur la période 2023 ?",
            n_results = 4
        )


load_dotenv()

api_key = os.getenv("API_KEY")

model = "mistral-tiny"

client = MistralClient(api_key=api_key)

st.title("Streamlit Question Answering App 🦜 🦚")

question_input = st.text_input("Question:")
if question_input:
    messages = [
        ChatMessage(role="user", content=f"{question_input} :{dbresults.get('documents')[0][0]} ?")
    ]

    chat_response = client.chat(
        model=model,
        messages=messages,
        max_tokens=100
    )
    
    st.text_area("Answer:", chat_response.choices[0].message.content)