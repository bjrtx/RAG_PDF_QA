import pytest
from unittest import mock
from unittest.mock import patch, MagicMock
from mistralai.client import MistralClient
from src.utils import Mistral_API, load_dir, get_answer, parse_PDF


def test_Mistral_API_success():
    with patch('os.getenv', return_value="valid_api_key"), \
         patch('mistralai.client.MistralClient') as MockClient:
        MockClient.return_value = MockClient
        model, client = Mistral_API()
        assert model == "mistral-tiny"
        assert isinstance(client, MistralClient)


def test_Mistral_API_failure():
    with mock.patch('os.getenv', return_value="fake_api_key"):
        model, client = Mistral_API()
        assert model is not None
        assert client is not None


def test_load_dir_success():
    # Simulate document objects that would be returned by
    # SimpleDirectoryReader
    mock_document1 = {"file_name": "doc1.pdf", "content": "PDF 1 content"}
    mock_document2 = {"file_name": "doc2.pdf", "content": "PDF 2 content"}

    with patch('src.utils.SimpleDirectoryReader.load_data',
               return_value=[mock_document1, mock_document2]):
        documents = load_dir()
        assert len(documents) == 2
        assert documents[0] == mock_document1
        assert documents[1] == mock_document2


def test_load_dir_failure():
    with patch('llama_index.SimpleDirectoryReader.load_data', return_value=[]):
        with pytest.raises(ValueError) as excinfo:
            load_dir()
        assert "No documents found in './data'" in str(excinfo.value)


def test_parse_PDF_success():
    mock_document1 = MagicMock()
    mock_document2 = MagicMock()
    documents = [mock_document1, mock_document2]

    mock_node1 = MagicMock()
    mock_node2 = MagicMock()
    expected_nodes = [mock_node1, mock_node2]

    with patch('src.utils.SimpleNodeParser.get_nodes_from_documents',
               return_value=expected_nodes):
        nodes = parse_PDF(documents)
        assert len(nodes) == 2
        assert nodes == expected_nodes


def test_get_answer_success():
    mock_collection = mock.MagicMock()
    mock_model = "mistral-tiny"
    mock_client = mock.MagicMock()
    mock_prompt_key = "RAG_PDF"

    mock_collection.query.return_value = {
        "documents": [["Simulated response"]],
        "metadatas": [[{"source": "source info"}]]
    }
    mock_response = mock.MagicMock()
    mock_response.choices = [mock.MagicMock()]
    mock_response.choices[0].message.content = "Expected answer"
    mock_client.chat.return_value = mock_response

    response = get_answer(
        "Test question",
        collection=mock_collection,
        model=mock_model,
        client=mock_client,
        prompt_key=mock_prompt_key
    )
    assert response == "Expected answer"
