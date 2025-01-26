import json

from bert_score import score


def calculate_bert_score(summary: str, original_text: str) -> float:
    P, R, F1 = score([summary], [original_text], lang="en", verbose=False)
    print(P, R, F1)
    return F1.item()


def test_summarize_pages():
    with open("data/full_texts.json", "r", encoding="utf-8") as f:
        full_texts = json.load(f)

    with open("data/summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)

    assert set(summaries.keys()) == set(full_texts.keys())

    for url, summary in summaries.items():
        original_text = full_texts[url]

        ratio = len(summary) / len(original_text)
        assert ratio <= 0.3, f"Summary length ratio {ratio} for {url} is out of bounds"

        bert_score = calculate_bert_score(summary, original_text)
        print(f"BERT score {bert_score} for {url}")
        assert bert_score >= 0.4, f"BERTScore {bert_score} for {url} is too low"
