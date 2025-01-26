import json
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


def parse_courses() -> List[str]:
    BASE_URL = "https://karpov.courses/"
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    courses = soup.find_all("a", class_="t978__innermenu-link")
    courses = [c.get("href") for c in courses]
    return courses


def extract_pages(hrefs: List[str]) -> Dict[str, str]:
    data_dict = {}
    for href in hrefs:
        text = BeautifulSoup(requests.get(href).text).text
        data_dict[href] = text

    with open("data/full_texts.json", "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    return data_dict


if __name__ == "__main__":
    courses = parse_courses()
    extract_pages(courses)
