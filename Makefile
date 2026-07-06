.PHONY: run lint format
run:
	uv run uvicorn api.main:app --reload

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .
