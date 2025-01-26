.PHONY: setup tests


setup:
	python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

tests:
	.venv/bin/pytest tests

run:
	streamlit run steps/step5_streamlit_service.py