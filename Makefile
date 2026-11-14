mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))
venv := $(mkfile_dir).venv/bin

.PHONY: all
all: clean dev

## remove all directories and files ignored through .gitignore
.PHONY: clean
clean:
	git clean -fXd

.PHONY: lint
lint: check-black check-complexity check-isort check-ruff mypy

.PHONY: check-complexity
check-complexity:
	(test -f $(venv)/flake8 || make dev)
	uv run flake8 .

.PHONY: check-black
check-black:
	(test -f $(venv)/black || make dev)
	uv run black --check her2_stopping_rule tests

.PHONY: black
black:
	(test -f $(venv)/black || make dev)
	uv run black her2_stopping_rule tests

.PHONY: check-isort
check-isort:
	(test -f $(venv)/isort || make dev)
	uv run isort --check her2_stopping_rule tests

.PHONY: isort
isort:
	(test -f $(venv)/isort || make dev)
	uv run isort her2_stopping_rule tests

.PHONY: check-ruff
check-ruff:
	(test -f $(venv)/ruff || make dev)
	uv run ruff check .

.PHONY: fix-ruff
fix-ruff:
	(test -f $(venv)/ruff || make dev)
	uv run ruff --fix .

.PHONY: mypy
mypy:
	(test -f $(venv)/mypy || make dev)
	uv run mypy her2_stopping_rule tests

.PHONY: test
test:
	(test -f $(venv)/pytest || make dev)
	uv run pytest

.PHONY: dev
dev:
	uv sync

.PHONY: install
install:
	uv sync --no-dev
	uv pip install .

.PHONY: wheel
wheel: clean
	(test -d .venv || make dev)
	uv build --sdist --wheel
