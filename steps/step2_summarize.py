import json
import os
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.proxyapi.ru/openai/v1",
)


def summarize_text(text_to_summarize: str) -> str:
    with open("prompts/summarize_prompt.txt", "r") as f:
        prompt = f.read()

    chat_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt.format(text_to_summarize=text_to_summarize),
            }
        ],
    )
    response_text = chat_completion.choices[0].message.content
    return response_text


def summarize_pages() -> Dict[str, str]:
    with open("data/full_texts.json", "r", encoding="utf-8") as f:
        full_texts = json.load(f)

    summaries = {}

    for url, text in tqdm(full_texts.items()):
        summary = summarize_text(text)
        summaries[url] = summary

    with open("data/summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries


if __name__ == "__main__":
    if not os.path.exists("data/summaries.json"):
        summarize_pages()
    else:
        print("Summaries already exist")
