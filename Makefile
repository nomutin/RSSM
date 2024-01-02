clean:
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache wandb && rm -f .coverage coverage.xml && find . -type d -name __pycache__ -exec rm -r {} +

install:
	if command -v rye >/dev/null 2>&1; then \
		rye sync --no-lock; \
	elif command -v pipenv >/dev/null 2>&1; then \
		pipenv install -r requirements-dev.lock --skip-lock; \
	else \
		pip install -r requirements-dev.lock; \
	fi


isort-format:
	isort .

isort-check:
	isort --check-only .

black-format:
	black .

black-check:
	black src/ --check 

ruff-format:
	ruff --fix .

ruff-check:
	ruff src/

format: isort-format black-format ruff-format

lint: isort-check black-check ruff-check


.PHONY: clean install format lint
