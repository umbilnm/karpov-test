import json
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np

from .step3_create_index import embed_text


@dataclass
class Document:
    """Представляет документ с его URL, текстом summary и эмбеддингом"""

    url: str
    summary: str
    embedding: List[float]


def load_index(
    index_path: str = "data/index.faiss", documents_path: str = "data/documents.json"
) -> tuple[faiss.Index, List[Document]]:
    """Загружает FAISS индекс и документы"""

    index = faiss.read_index(index_path)

    with open(documents_path, "r", encoding="utf-8") as f:
        documents_data = json.load(f)
    documents = [Document(**doc) for doc in documents_data]

    return index, documents


def retrieve_similar(query: str, top_k: int = 3) -> List[tuple[str, str]]:
    """
    Находит top_k наиболее похожих документов для заданного текста

    Args:
        query: Текст запроса
        top_k: Количество похожих документов для возврата

    Returns:
        Список из top_k кортежей (url, summary) наиболее похожих документов
    """
    index, documents = load_index()
    query_embedding = embed_text(query)

    distances, indices = index.search(
        np.array([query_embedding], dtype=np.float32), top_k
    )

    similar_documents = [
        (documents[idx].url, documents[idx].summary) for idx in indices[0]
    ]
    return similar_documents
