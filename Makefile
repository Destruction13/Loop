.PHONY: run lint test

run:
	python -m app.main

lint:
	python -m compileall app

test:
	pytest
