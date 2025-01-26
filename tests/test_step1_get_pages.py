from unittest.mock import Mock, mock_open, patch

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
    with patch("requests.get", return_value=mock_response) as mock_get:
        courses = parse_courses()
        assert isinstance(courses, list)
        assert len(courses) == 2
        assert "https://karpov.courses/analytics" in courses
        assert "https://karpov.courses/python" in courses
        mock_get.assert_called_once_with("https://karpov.courses/")


def test_extract_pages_error_response():
    test_hrefs = ["https://karpov.courses/error"]

    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            extract_pages(test_hrefs)

        assert "Failed to fetch https://karpov.courses/error" in str(exc_info.value)


def test_extract_pages():
    test_hrefs = ["https://karpov.courses/test1", "https://karpov.courses/test2"]
    mock_file = mock_open()

    with patch("requests.get") as mock_get, patch("builtins.open", mock_file):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Test content"
        mock_get.return_value = mock_response

        result = extract_pages(test_hrefs)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert all(url in result for url in test_hrefs)

    handle = mock_file()
    written_data = "".join(call.args[0] for call in handle.write.call_args_list)

    assert "https://karpov.courses/test1" in written_data
    assert "https://karpov.courses/test2" in written_data
    assert "Test content" in written_data
