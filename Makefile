lint:
	uv run ruff check --fix && \
	uv run ruff format

check:
	uv run ty check && \
	uv run ruff check && \
	uv run ruff format --check

test:
	uv run pytest tests

build: lint check test
	rm -rf dist && \
	uv build
