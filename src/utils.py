from typing import List, Tuple, Optional, Match
from io import BytesIO
import base64
import os
import re

import chromadb
from chromadb.api.models.Collection import Collection
import streamlit as st
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import Document, BaseNode
from llama_index import SimpleDirectoryReader
from pypdf import PdfReader
from dotenv import load_dotenv
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException

from src.prompts import PROMPTS


def prepare_data_for_mistral(
    uploaded_file: Optional[BytesIO] = None,
    use_dir: bool = False,
    include_collection: bool = True,
) -> Tuple[
    List[Document], Optional[List[BaseNode]], Optional[Collection], str, MistralClient
]:
    """
    This unified method aims to prepare data to send it to the Mistral API
    """
    if use_dir:
        documents = load_dir()
    elif uploaded_file is not None:
        documents = load_pdf(uploaded_file)
    else:
        documents = None

    if include_collection:
        nodes = parse_pdf(documents)
        collection = vector_db(nodes)
    else:
        nodes = collection = None

    model, client = mistral_api()

    return documents, nodes, collection, model, client


def upload_pdf() -> Optional[BytesIO]:
    return st.file_uploader("Upload a PDF", type="pdf")


def display_pdf(uploaded_file: BytesIO) -> None:
    """Display uploaded PDF in streamlit."""
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert to utf-8
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")
    # Embed PDF in HTML
    width = 800
    pdf_display = (
        f"<iframe src="
        f'"data:application/pdf;base64,{base64_pdf}#toolbar=0"'
        f"width={str(width)} height={str(width * 4 / 3)}"
        'type="application/pdf"></iframe>'
    )
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


@st.cache_resource()
def load_pdf(uploaded_file: BytesIO) -> List:
    """Load uploaded PDF file into one document."""
    pdf = PdfReader(uploaded_file)
    documents = []
    text = ""
    filename = f"{uploaded_file.name}"
    for label, page in enumerate(pdf.pages):
        text += page.extract_text()
        documents.append(
            Document(text=text, extra_info={"page_label": label, "file_name": filename})
        )
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


def parse_pdf(documents: List) -> List[BaseNode]:
    """Parse PDF into nodes."""
    node_parser = SimpleNodeParser.from_defaults(chunk_size=5000)
    return node_parser.get_nodes_from_documents(documents)


def vector_db(nodes: List[BaseNode]) -> Collection:
    """Create and embed PDF in a vector database."""
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name="test", metadata={"hnsw:space": "cosine"}
        )
        for i, node in enumerate(nodes):
            collection.add(
                documents=[node.get_content()],
                metadatas=[{"source": node.get_metadata_str()}],
                ids=[f"{i}"],
            )
        return collection
    except Exception as e:
        st.error(f"Error setting up the vector database: {str(e)}")
        raise


@st.cache_resource()
def mistral_api() -> Tuple[str, MistralClient]:
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


def get_summary(docs: List, client: MistralClient, model: str) -> str:
    """Receive the PDF uploaded and make a summary"""
    messages = [
        ChatMessage(
            role="user",
            content=f"Make a summary written in the third person plural 'they' "
            f"of the following scientific paper PDF:"
            f"{docs[0].text} and write it in the following form: the title, "
            f"the authors, an abstract, the main contribution, "
            f"the key findings, and a conclusion.",
        )
    ]
    try:
        chat_response = client.chat(model=model, messages=messages)
        return chat_response.choices[0].message.content
    except MistralAPIException as e:
        if e.http_status == 400:
            st.error(
                "File too big for the Mistral context window (32k tokens)."
                "Please try with a smaller file."
            )
        else:
            st.error(f"An error occurred: {e.message}")
        return ""


def get_answer(
    question_input: str,
    collection: Collection,
    model: str,
    client: MistralClient,
    prompt_key: str,
) -> str:
    """
    Question answering function based on Retrieved information (chunk of PDF)
    """
    try:
        db_results = collection.query(query_texts=[question_input])
        if db_results is not None:
            documents = db_results.get("documents")
            metadata = db_results.get("metadatas")
            if documents and metadata is not None:
                content = documents[0][0]
                source = metadata[0][0]["source"]
            else:
                content = source = None
            page_number_match: Optional[Match[str]] = re.search(
                r"page_label: (\d+)", str(source)
            )
            filename_match: Optional[Match[str]] = re.search(
                r"file_name: (.+)", str(source)
            )

            page_number = page_number_match.group(1) if page_number_match else "Unknown"
            filename = filename_match.group(1) if filename_match else "Unknown"

            prompt_template = PROMPTS[prompt_key]
            prompt = prompt_template.format(
                question=question_input,
                content=content,
                filename=filename,
                page_number=page_number,
            )

            messages = [ChatMessage(role="user", content=prompt)]
            chat_response = client.chat(model=model, messages=messages)
            answer = chat_response.choices[0].message.content
        else:
            answer = ""
        return answer
    except Exception as e:
        st.error(f"Failed to get answer: {str(e)}")
        raise
