.PHONY: install dev test lint seed demo serve-mcp serve-mcp-stdio serve-a2a benchmark docker-up docker-down clean

install:
	pip install -e ".[protocols]"

dev:
	pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src tests examples

seed:
	anthromesh init
	anthromesh seed --decisions examples/sample_decisions.json --cohort examples/sample_cohort.json

demo:
	python examples/demo_end_to_end.py

benchmark:
	anthromesh benchmark

serve-mcp:
	anthromesh serve-mcp --transport http --port 8081

serve-mcp-stdio:
	anthromesh serve-mcp --transport stdio

serve-a2a:
	anthromesh serve-a2a --port 8082

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache anthromesh_vault.db
