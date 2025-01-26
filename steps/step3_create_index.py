import json
import os
from typing import Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from step1_get_pages import extract_pages, parse_courses
from step2_summarize import summarize_pages

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
model = "text-embedding-ada-002"
client = OpenAI(
    base_url="https://api.proxyapi.ru/openai/v1",
)


def embed_text(text: str):
    response = client.embeddings.create(input=text, model=model)
    embedding = response.data[0].embedding
    return embedding


def create_index(
    summaries: Dict[str, str]
) -> Tuple[faiss.Index, List[str], np.ndarray]:

    urls = list(summaries.keys())
    texts = list(summaries.values())
    embeddings = np.array([embed_text(text) for text in texts], dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, urls, embeddings


def save_index(
    index: faiss.Index,
    urls: List[str],
    embeddings: np.ndarray,
    summaries: Dict[str, str],
    index_path: str = "data/index.faiss",
    documents_path: str = "data/documents.json",
):
    faiss.write_index(index, index_path)

    documents_data = [
        {"url": url, "summary": summaries[url], "embedding": emb.tolist()}
        for url, emb in zip(urls, embeddings)
    ]

    with open(documents_path, "w", encoding="utf-8") as f:
        json.dump(documents_data, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.exists("data/summaries.json"):
        courses = parse_courses()
        pages_content = extract_pages(courses)
        summarize_pages(pages_content)

    with open("data/summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)

    index, urls, embeddings = create_index(summaries)
    save_index(index, urls, embeddings, summaries)


if __name__ == "__main__":
    main()
