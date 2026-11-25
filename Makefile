PYTHON		= uv run python3
pytest		= uv run pytest
UV			= uv
VENV		= .venv
RAG			= -m src


TMP_DIRS	= __pycache__ .mypy_cache .ruff_cache build dist pac-man-linux.spec
DATASET     = data/datasets/$(SCOPE)/dataset_$(NAME)_public.json
TRUTH       = data/datasets/AnsweredQuestions/dataset_$(NAME)_public.json
SEARCH_DIR  = data/output/search_results/$(SCOPE)
ANSWER_DIR  = data/output/search_results_and_answer/$(SCOPE)
RESULTS     = $(SEARCH_DIR)/dataset_$(NAME)_public.json

CONFIG 		?= ""
K          	?= 10
CHUNK     	?= 2000
QUERY   	?= How to configure the OpenAI server?
SCOPE      	?= UnansweredQuestions
NAME       	?= docs
MOULI      	?= ./moulinette/moulinette-ubuntu


index:
	$(PYTHON) $(RAG) index --max_chunk_size $(CHUNK)

search:
	$(PYTHON) $(RAG) search "$(QUERY)" --k $(K)

search_dataset:
	$(PYTHON) $(RAG) search_dataset --dataset_path $(DATASET) --k $(K) \
		--save_directory $(SEARCH_DIR)

answer:
	$(PYTHON) $(RAG) answer "$(QUERY)" --k $(K)

answer_dataset:
	$(PYTHON) $(RAG) answer_dataset --student_search_results_path $(RESULTS) \
		--save_directory $(ANSWER_DIR)

evaluate:
	$(PYTHON) $(RAG) evaluate --student_search_results_path $(RESULTS) \
		--dataset_path $(TRUTH)

score:
	$(MOULI) evaluate_student_search_results $(RESULTS) $(TRUTH) \
		--k $(K) --max_context_length 2000


install:
	@echo ">>> Installation de uv..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo ">>> Sync des dépendances (prod + dev)..."
	$(UV) sync
	@echo ">>> OK — projet prêt !"


test-finder:
	@$(PYTHON) -m pytest src/test/test_finder.py -v

test-chunker:
	@$(PYTHON) -m pytest src/test/test_chunker.py -v

test-parser:
	@$(PYTHON) -m pytest src/test/test_pars.py -v

test-evaluate:
	@$(PYTHON) -m pytest src/test/test_evaluate.py -v

test: test-finder test-chunker test-parser test-evaluate


run:
	@echo ">>> Lancement de la simulation..."
	@$(PYTHON) $(RAG) $(CONFIG)

debug:
	@$(PYTHON) -m pdb -m $(RAG)

lint:
	@$(PYTHON) -m flake8 --max-line-length=79 --exclude=.venv,data
	@$(PYTHON) -m mypy --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/


lint-strict:
	@$(PYTHON) -m flake8 --max-line-length=79 --exclude=.venv,data
	@$(PYTHON) -m mypy --explicit-package-bases --strict src/

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

.PHONY: install run debug clean lint lint-strict test index search \
	search_dataset answer answer_dataset evaluate score