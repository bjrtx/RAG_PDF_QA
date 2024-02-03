import pytest
from unittest import mock
from unittest.mock import patch, MagicMock
from mistralai.client import MistralClient
from main import Mistral_llm, load_data, get_answer


def test_Mistral_llm_success():
    with patch('os.getenv', return_value="valid_api_key"), \
         patch('mistralai.client.MistralClient') as MockClient:
        MockClient.return_value = MockClient
        model, client = Mistral_llm()
        assert model == "mistral-tiny"
        assert isinstance(client, MistralClient)


def test_Mistral_llm():
    with mock.patch('os.getenv', return_value="fake_api_key"):
        model, client = Mistral_llm()
        assert model is not None
        assert client is not None


def test_load_data_success():
    # Simulate document objects that would be returned by
    # SimpleDirectoryReader
    mock_document1 = MagicMock(name='Document1')
    mock_document2 = MagicMock(name='Document2')

    # Simulate nodes created from PDF documents
    mock_node1 = MagicMock(id='doc1', content='PDF 1 content')
    mock_node2 = MagicMock(id='doc2', content='PDF 2 content')
    expected_nodes = [mock_node1, mock_node2]

    with patch('llama_index.SimpleDirectoryReader.load_data',
               return_value=[mock_document1, mock_document2]), \
         patch('llama_index.node_parser.SimpleNodeParser.get_nodes_from_documents',
               return_value=expected_nodes):
        base_nodes = load_data()
        assert len(base_nodes) == 2
        assert base_nodes[0].id == 'doc1'
        assert base_nodes[0].content == 'PDF 1 content'
        assert base_nodes[1].id == 'doc2'
        assert base_nodes[1].content == 'PDF 2 content'


def test_load_data():
    with patch('llama_index.SimpleDirectoryReader.load_data', return_value=[]):
        with pytest.raises(ValueError) as excinfo:
            load_data()
        assert "No documents found in './data'" in str(excinfo.value)


def test_get_answer_success():
    with patch('main.vectorDB') as mock_vectorDB, \
         patch('main.Mistral_llm') as mock_Mistral_llm:
        mock_vectorDB.return_value.query.return_value = {
            "documents": [["Simulated response"]]
        }
        mock_MistralClient = mock.Mock()
        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message.content = "Expected answer"
        mock_MistralClient.chat.return_value = mock_response
        mock_Mistral_llm.return_value = ("mistral-tiny", mock_MistralClient)
        response = get_answer("Test question")
        assert response == "Expected answer"


def test_get_answer():
    with mock.patch('main.vectorDB') as mock_vectorDB, \
         mock.patch('main.Mistral_llm') as mock_Mistral_llm:
        mock_vectorDB.return_value.query.return_value = {
            "documents": [["Simulated response"]]
        }
        mock_MistralClient = mock.Mock()
        mock_MistralClient.chat.return_value.choices = [mock.Mock()]
        mock_MistralClient.chat.return_value.choices[0].message.content = \
            "Simulated response"
        mock_Mistral_llm.return_value = ("mistral-tiny", mock_MistralClient)
        response = get_answer("Test question")
        assert response == "Simulated response"
