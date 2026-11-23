.PHONY: install data test lint serve demo

install:
	pip install -e ".[dev,deep]"

data:
	python examples/generate_datasets.py

test:
	AGENTML_PROVIDER=mock pytest

lint:
	ruff check .

serve:
	agentml serve

demo: data
	agentml demo
