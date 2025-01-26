import json
import os
import time
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
        time.sleep(5)
        response = requests.get(href)
        if response.status_code == 200:
            text = BeautifulSoup(response.text, "html.parser").text
            data_dict[href] = text
        else:
            raise Exception(f"Failed to fetch {href}, response: {response.status_code}")

    with open("data/full_texts.json", "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    return data_dict


if __name__ == "__main__":
    if not os.path.exists("data/full_texts.json"):
        courses = parse_courses()
        extract_pages(courses)
    else:
        print("Data already exists")
