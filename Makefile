.PHONY: run
run:
	uv run uvicorn api.main:app --reload
