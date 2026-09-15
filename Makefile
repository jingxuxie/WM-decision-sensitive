PYTHON ?= python
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MPLCONFIGDIR ?= /tmp/wm-decision-sensitive-matplotlib

.PHONY: all test experiments figures paper clean
all: test experiments figures paper
test:
	$(PYTHON) -m pytest -q
experiments:
	$(PYTHON) experiments/run_experiments.py
figures:
	$(PYTHON) experiments/make_figures.py
paper:
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex
clean:
	rm -f paper/main.aux paper/main.log paper/main.out paper/build.log
