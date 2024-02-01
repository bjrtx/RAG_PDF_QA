import pytest
from unittest import mock
from main import Mistral_llm, load_data, get_answer


def test_Mistral_llm():
    model, client = Mistral_llm()
    assert model is not None
    assert client is not None


def test_load_data():
    base_nodes = load_data()
    assert isinstance(base_nodes, list)


def test_get_answer():
    # Use a mock or stub to simulate the behaviour of dependencies ?
    response = get_answer("test question")
    print(response)
    assert isinstance(response, str)


# Integration test
def test_full_interaction():
    with mock.patch('main.vectorDB') as mock_vectorDB:
        mock_vectorDB.return_value.query.return_value = {
            "documents": [["Simulated response"]]
            }

        with mock.patch('main.Mistral_llm') as mock_Mistral_llm:
            mock_MistralClient = mock.Mock()
            mock_MistralClient.chat.return_value.choices = [mock.Mock()]
            mock_MistralClient.chat.return_value.choices[0].message.content = "Simulated response"
            mock_Mistral_llm.return_value = (
                "mistral-tiny",
                mock_MistralClient
                )

            response = get_answer("Test question")
            assert response == "Simulated response"
