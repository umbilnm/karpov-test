import json
from unittest.mock import mock_open, patch

import faiss
import numpy as np
import pytest

from steps.step4_retrieval import Document, load_index, retrieve_similar


@pytest.fixture
def mock_index_and_documents():
    documents = [
        Document(url="url1", summary="summary1", embedding=[0.1, 0.2]),
        Document(url="url2", summary="summary2", embedding=[0.3, 0.4]),
    ]
    index = faiss.IndexFlatL2(2)
    return index, documents


def test_retrieve_similar(mock_index_and_documents):
    index, documents = mock_index_and_documents
    with (
        patch("steps.step4_retrieval.load_index", return_value=(index, documents)),
        patch("steps.step4_retrieval.embed_text", return_value=np.array([0.1, 0.2])),
    ):
        results = retrieve_similar("test query", top_k=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, tuple) for r in results)


def test_load_index():
    # Mock data for documents.json
    mock_documents_data = json.dumps(
        [
            {
                "url": "http://example.com",
                "summary": "Example summary",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]
    )

    mock_index = faiss.IndexFlatL2(3)

    with (
        patch("builtins.open", mock_open(read_data=mock_documents_data)),
        patch("faiss.read_index", return_value=mock_index),
    ):

        index, documents = load_index()

        assert index == mock_index

        assert len(documents) == 1
        assert documents[0].url == "http://example.com"
        assert documents[0].summary == "Example summary"
        assert documents[0].embedding == [0.1, 0.2, 0.3]
