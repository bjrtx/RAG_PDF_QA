import pytest
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