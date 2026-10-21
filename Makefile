PYTHON		= uv run python3
pytest		= uv run pytest
UV			= uv
VENV		= .venv
RAG			= RAG.py
TMP_DIRS	= __pycache__ .mypy_cache .ruff_cache build dist pac-man-linux.spec

CONFIG ?= ""

install:
	@echo ">>> Installation de uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo ">>> Sync des dépendances (prod + dev)..."
	$(UV) sync
	@echo ">>> OK — projet prêt !"



test-finder:
	@$(PYTHON) -m pytest test/test_finder.py -v

test-chunker:
	@$(PYTHON) -m pytest test/test_chunker.py -v

test-parser:
	@$(PYTHON) -m pytest test/test_pars.py -v

test: test-finder test-chunker test-parser



run:
	@echo ">>> Lancement de la simulation..."
	@$(PYTHON) $(RAG) $(CONFIG)

debug:
	@$(PYTHON) -m pdb -m $(RAG)

lint:
	@$(PYTHON) -m flake8 --max-line-length=79 --exclude=.venv,Rag_subject
	@$(PYTHON) -m mypy --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs pars/
	@$(PYTHON) -m mypy --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs RAG.py


lint-strict:
	@$(PYTHON) -m flake8 --max-line-length=79 --exclude=.venv,Rag_subject
	@$(PYTHON) -m mypy --explicit-package-bases --strict pars/
	@$(PYTHON) -m mypy --explicit-package-bases --strict RAG.py

clean:
	@echo ">>> Suppression des fichiers temporaires..."
	@rm -rf $(TMP_DIRS)
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	@echo ">>> Clean OK !"

fclean: clean
	@echo ">>> Suppression du venv..."
	@rm -rf $(VENV)
	@echo ">>> FClean OK !"


.PHONY: run install debug clean fclean lint lint-strict test