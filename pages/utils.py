import chromadb
from chromadb.api.models.Collection import Collection
import streamlit as st
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import Document, BaseNode
from pypdf import PdfReader
from typing import List, Tuple
from io import BytesIO
import base64
from dotenv import load_dotenv
import os
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException


def display_pdf(uploaded_file: BytesIO) -> None:
    """Display uploaded PDF in streamlit."""
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    # Embed PDF in HTML
    width = 800
    pdf_display = f'<iframe src=' \
        f'"data:application/pdf;base64,{base64_pdf}#toolbar=0"' \
        f'width={str(width)} height={str(width*4/3)}' \
        'type="application/pdf"></iframe>'
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_PDF(uploaded_file: BytesIO) -> List:
    """Load uploaded PDF file into one document."""
    pdf = PdfReader(uploaded_file)
    docs = []
    text = ""
    metadata = {"file_name": uploaded_file.name}
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    docs.append(Document(metadata=metadata, text=text))
    # if >32k tokens => hierarchical summarization
    return docs


@st.cache_resource()
class Embeddings():
    def __init__(self, uploaded_file: BytesIO):
        self.uploaded_file = uploaded_file

    def parse_PDF(self) -> List[BaseNode]:
        """Parse uploaded PDF file into nodes."""
        pdf = PdfReader(self.uploaded_file)
        docs = []
        text = ""
        metadata = {"file_name": self.uploaded_file.name}
        for page in range(len(pdf.pages)):
            text += pdf.pages[page].extract_text()
        docs.append(Document(metadata=metadata, text=text))
        node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
        self.nodes = node_parser.get_nodes_from_documents(docs)
        return self.nodes

    def vectorDB(self) -> Collection:
        """Create and embed pdf in a vector database."""
        try:
            chroma_client = chromadb.Client()
            collection = chroma_client.get_or_create_collection(
                name="test",
                metadata={"hnsw:space": "cosine"})
            for i, node in enumerate(self.nodes):
                collection.add(
                    documents=[node.get_text()],
                    metadatas=[{'source': node.metadata.get('file_name')}],
                    ids=[f'{i}'])
            return collection
        except Exception as e:
            st.error(f"Error setting up the vector database: {str(e)}")
            raise


@st.cache_resource()
def Mistral_API() -> Tuple[str, MistralClient]:
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key for Mistral is missing")
    model = "mistral-tiny"
    try:
        client = MistralClient(api_key=api_key)
        return model, client
    except Exception as e:
        st.error(f"Failed to initialize Mistral Client: {str(e)}")
        raise


def summary(
        docs: List,
        client: MistralClient,
        model: str) -> str:
    messages = [
        ChatMessage(
            role="user",
            content=f"Make a summary written in the third person plural 'they'"
            f"of the following scientific paper PDF:"
            f"{docs[0].text} and write it in the following form: the title,"
            f"the authors, an abstract, the main contributionn,"
            f"the key findings, and a conclusion."
            )
            ]
    try:
        chat_response = client.chat(model=model, messages=messages)
        return chat_response.choices[0].message.content
    except MistralAPIException as e:
        if e.http_status == 400:
            st.error("File too big for the Mistral context window 32k tokens."
                     "Please try with a smaller file.")
            return ""
        else:
            error_message = f"An error occurred: {e.message}"
            st.error(error_message)
            return ""


def get_answer(
        question_input: str,
        collection: Collection,
        model: str,
        client: MistralClient) -> str:
    try:
        dbresults = collection.query(query_texts=[question_input])
        if dbresults and dbresults.get('documents'):
            content = dbresults.get('documents')[0][0]
            messages = [
                ChatMessage(
                    role="user",
                    content=f"{question_input}:{content} ?")
                    ]
            chat_response = client.chat(model=model, messages=messages)
            return chat_response.choices[0].message.content
        else:
            return "No results found in the database."
    except Exception as e:
        st.error(f"Failed to get answer: {str(e)}")
        raise
