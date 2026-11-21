SHELL := /bin/bash
VENV := .venv/bin/activate

.PHONY: venv install train run test docker-build docker-run

venv:
	@if [ ! -f "$(VENV)" ]; then \
		echo "Création de .venv..."; \
		python3 -m venv .venv; \
		. $(VENV) && pip install -r requirements.txt; \
	fi

install: venv
	. $(VENV) && pip install -r requirements.txt

train: venv
	. $(VENV) && python3 src/train.py

run: venv
	. $(VENV) && gunicorn api.server:app --bind 0.0.0.0:5000 --workers 2 --threads 4

test: venv
	. $(VENV) && pip install pytest && . $(VENV) && python3 -m pytest tests/ -v

docker-build:
	docker build -t logistic-regression-api .

docker-run:
	docker run -d -p 5000:5000 --name heart-api logistic-regression-api

ui: venv
	. $(VENV) && streamlit run ui/app.py --server.port 8501
