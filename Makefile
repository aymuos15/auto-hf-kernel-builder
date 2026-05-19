# Local-only. No ssh/Spark, no kernel_lib. One entrypoint: run.py.
.PHONY: help prepare run clean

LEVEL ?= 3
PROBLEM ?= 4

help:
	@echo "make prepare LEVEL=3 PROBLEM=4   # freeze the contract only"
	@echo "make run     LEVEL=3 PROBLEM=4   # prepare -> code loop -> report"
	@echo "make clean                       # drop generated kernels/results"

prepare:
	python3 core/prepare.py --level $(LEVEL) --problem $(PROBLEM)

run:
	python3 run.py --level $(LEVEL) --problem $(PROBLEM)

clean:
	find tasks -name kernel.py -delete -o -name result.json -delete -o -name .best_kernel.py -delete 2>/dev/null || true
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
