from unittest.mock import mock_open, patch

import faiss
import numpy as np
import pytest

from steps.step3_create_index import create_index, save_index


@pytest.fixture
def sample_summaries():
    return {"url1": "summary1", "url2": "summary2"}


def test_create_index(sample_summaries):
    with patch(
        "steps.step3_create_index.embed_text", return_value=np.array([0.1, 0.2])
    ) as mock_embed_text:
        index, urls, embeddings = create_index(sample_summaries)
        assert isinstance(index, faiss.Index)
        assert isinstance(urls, list)
        assert isinstance(embeddings, np.ndarray)
        assert len(urls) == len(sample_summaries)
        mock_embed_text.assert_any_call("summary1")
        mock_embed_text.assert_any_call("summary2")


def test_save_index(sample_summaries):
    mock_index = faiss.IndexFlatL2(2)
    urls = ["url1", "url2"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch("faiss.write_index"), patch("builtins.open", mock_open()):
        save_index(mock_index, urls, embeddings, sample_summaries)
