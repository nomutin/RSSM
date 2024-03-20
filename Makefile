help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## 実行に影響のないファイル(.*_cacheとか)を削除
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/ wandb/ && \
	rm -f .coverage coverage.xml *.out && \
	find . -type d -name __pycache__ -exec rm -r {} +

format:  ## コードのフォーマット(isort->black->ruff)
	isort . && \
	black . && \
	ruff format .

lint:  ## コードのLint(isort->black->mypy->ruff)
	isort . --check && \
	black . --check && \
	mypy . && \
	ruff check .

test:  ## テストを実行(カバレッジも計測)
	pytest -ra --cov=src --cov-report=term --cov-report=xml

install:  ## 仮想環境の作成
	rye sync --no-lock && \
	rye run pre-commit install

train:  ## `make train f=<path_to_config>` で学習を実行
	python scripts/train.py fit --config $(f)

debug: ## `make debug f=<path_to_config>` でデバック用epochを回す
	python scripts/train.py fit --config $(f) --trainer.fast_dev_run true

download: ## `make download l=<url>` でgoogle driveからダウンロード
	python scripts/download.py $(l)

.PHONY: help clean format lint test install train debug download