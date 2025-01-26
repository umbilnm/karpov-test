from unittest.mock import patch

import faiss
import numpy as np
import pytest

from steps.step4_retrieval import Document, retrieve_similar


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
