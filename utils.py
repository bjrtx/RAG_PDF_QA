import chromadb
from chromadb.api.models.Collection import Collection
import streamlit as st
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import Document, BaseNode
from llama_index import SimpleDirectoryReader
from pypdf import PdfReader
from typing import List, Tuple, Optional
from io import BytesIO
import base64
from dotenv import load_dotenv
import os
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException


def prepare_data_for_mistral(
        uploaded_file: Optional[BytesIO] = None,
        use_dir: bool = False,
        include_collection: bool = True
        ) -> Tuple[
            List[Document],
            Optional[List[BaseNode]],
            Optional[Collection],
            str,
            MistralClient
            ]:
    """
    This unified method aims to prepare data to send it to the Mistral API
    """
    if use_dir:
        documents = load_dir()
    else:
        if uploaded_file is not None:
            documents = load_PDF(uploaded_file)

    collection = None
    nodes = None
    if include_collection:
        nodes = parse_PDF(documents)
        collection = vectorDB(nodes)

    model, client = Mistral_API()

    return documents, nodes, collection, model, client


def upload_pdf() -> Optional[BytesIO]:
    uploaded_file = st.file_uploader("Download PDF", type="pdf")
    if uploaded_file is not None:
        return uploaded_file
    else:
        return None


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


@st.cache_resource()
def load_PDF(uploaded_file: BytesIO) -> List:
    """Load uploaded PDF file into one document."""
    pdf = PdfReader(uploaded_file)
    documents = []
    text = ""
    metadata = {"file_name": uploaded_file.name}
    for page in range(len(pdf.pages)):
        text += pdf.pages[page].extract_text()
    documents.append(Document(text=text, extra_info=metadata))
    return documents


@st.cache_resource()
def load_dir() -> List:
    """Load PDF files in the data directory."""
    try:
        documents = SimpleDirectoryReader(input_dir="./data").load_data()
        if not documents:
            raise ValueError("No documents found in './data'")
        return documents
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise


def parse_PDF(documents: List) -> List[BaseNode]:
    """Parse PDF into nodes."""
    node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
    nodes = node_parser.get_nodes_from_documents(documents)
    return nodes


def vectorDB(nodes: List[BaseNode]) -> Collection:
    """Create and embed pdf in a vector database."""
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="test",
            metadata={"hnsw:space": "cosine"})
        for i, node in enumerate(nodes):
            collection.add(
                documents=[node.get_content()],
                metadatas=[
                    {'source': f'{node.get_metadata_str()}'}
                    ],
                ids=[f'{i}'])
        return collection
    except Exception as e:
        st.error(f"Error setting up the vector database: {str(e)}")
        raise


@st.cache_resource()
def Mistral_API() -> Tuple[str, MistralClient]:
    """Connect to the Mistral API"""
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


def get_summary(
        docs: List,
        client: MistralClient,
        model: str) -> str:
    """Receive the PDF uploaded and make a summary"""
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
    """
    Question answering function based on Retrieved information (chunk of PDF)
    """
    try:
        dbresults = collection.query(query_texts=[question_input])
        if dbresults is not None:
            documents = dbresults.get('documents')
            metadatas = dbresults.get('metadatas')
            if documents and metadatas is not None:
                content = documents[0][0]
                sourcename = metadatas[0][0]['source']
            messages = [
                ChatMessage(
                    role="user",
                    content=f"I want you to answer a question based on"
                    f"a chunk of a retrieved file that I will give you."
                    f"If you don't find the answer in"
                    f" the text that I give you, answer:"
                    f"'I don't find anything in the corresponding text'."
                    f"First write the filename from:{sourcename} and the"
                    f"page_label (int) from:{sourcename} if there is one,"
                    f"then Answer the question:{question_input} with the"
                    f"text {content}."
                    )
                    ]
            chat_response = client.chat(model=model, messages=messages)
            answer = chat_response.choices[0].message.content
        else:
            answer = ""
        return answer
    except Exception as e:
        st.error(f"Failed to get answer: {str(e)}")
        raise
