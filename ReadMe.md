# RAG Application on PDF files with Mistral API, ChromaDB and llama_index

## What is RAG Application ? 

To enable large language model to also have knowledge of data outside of its training data, e.g. company or research data, you can embed this data into a vector database and let an LLM retrieve the relevant documents and data. The LLM will then construct a coherent answer with the retrieved data. It enables you to connect pre-trained models to external, up-to-date information sources that can generate more accurate and more useful outputs.

## How does it works ?

The first steps in this process are as follows:
- Downloading PDFs.
- Cutting into chunks of a given size (max token).
- Embedding the chunks.
- Storing the embeddings in a vector database.
To do this, we use the `llama_index` library, which provides ready-to-use functions, and `chromaDB` vector database.

The following steps involve the use of LLM:
- Embedding of the question.
- Calculation of the closest similarities between the PDF embeddings and the question one.
- Finally, the question and the chunk are sent to the LLM to generate a response.
To do this, we're using the `streamlit` library, which provides a user interface based on python code, and the `Mistral API` to connect to the LLM.

## Requirements

The experiments were performed on a local laptop without GPU. In addition, Python version 3.10.11 and the dependencies in [requirements.txt](./requirements.txt) were used. These can be installed in a virtual environment with the following commands:
```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install upgrade pip
pip install -r requirements.txt
pip install -e .
```

**Note**: our experiments were conducted with the exact package versions specified in `requirements.txt`. Future updates may alter reproducibility. 

## Unit testing

To run unit tests, the user can add tests to the `test_app.py` file and run the command:
pytest test_app.py
```sh
pytest test_app.py
```

## Running

To use this repository, the user must copy the PDFs into the `data` folder and use the following command to execute the code: 
```sh
streamlit run main.py
```
This will generate an interface for asking questions and reading answers from the LLM.