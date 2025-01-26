import json
from unittest.mock import Mock, patch

from bert_score import score

from steps.step2_summarize import summarize_text


def calculate_bert_score(summary: str, original_text: str) -> float:
    P, R, F1 = score([summary], [original_text], lang="en", verbose=False)
    return F1.item()


def test_summarize_text():
    test_text = """
    Курс по Python для начинающих. 
    Научитесь основам программирования, работе с данными и базовым алгоритмам.
    Продолжительность курса 3 месяца. 
    Включает практические задания и проекты.
    """

    mock_response = Mock()
    mock_response.choices = [
        Mock(
            message=Mock(
                content="Краткое описание курса Python: базовое программирование, работа с данными, длительность 3 месяца."
            )
        )
    ]

    with patch(
        "steps.step2_summarize.client.chat.completions.create",
        return_value=mock_response,
    ) as mock_create:
        summary = summarize_text(test_text)
        assert (
            summary
            == "Краткое описание курса Python: базовое программирование, работа с данными, длительность 3 месяца."
        )
        mock_create.assert_called_once()


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
        assert bert_score >= 0.4, f"BERTScore {bert_score} for {url} is too low"
