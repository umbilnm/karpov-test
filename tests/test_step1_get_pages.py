from unittest.mock import Mock, patch

import pytest

from steps.step1_get_pages import extract_pages, parse_courses


@pytest.fixture
def mock_response():
    mock = Mock()
    mock.text = """
        <html>
            <a class="t978__innermenu-link" href="https://karpov.courses/analytics">Analytics</a>
            <a class="t978__innermenu-link" href="https://karpov.courses/python">Python</a>
        </html>
    """
    return mock


def test_parse_courses(mock_response):
    with patch("requests.get", return_value=mock_response):
        courses = parse_courses()
        assert isinstance(courses, list)
        assert len(courses) == 2
        assert "https://karpov.courses/analytics" in courses
        assert "https://karpov.courses/python" in courses


def test_extract_pages():
    test_hrefs = ["https://karpov.courses/test1", "https://karpov.courses/test2"]
    with patch("requests.get") as mock_get:
        mock_get.return_value.text = "Test content"
        result = extract_pages(test_hrefs)
        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(url in result for url in test_hrefs)
