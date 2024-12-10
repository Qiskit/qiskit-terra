# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

OS := $(shell uname -s)

.PHONY: default ruff env lint lint-incr style black test test_randomized pytest pytest_randomized test_ci coverage coverage_erase clean

default: ruff style lint-incr test ;

# Dependencies need to be installed on the Anaconda virtual environment.
env:
	if test $(findstring qiskitenv, $(shell conda info --envs | tr '[:upper:]' '[:lower:]')); then \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	else \
		conda create -y -n Qiskitenv python=3; \
		bash -c "source activate Qiskitenv;pip install -r requirements.txt"; \
	fi;

# Ignoring generated ones with .py extension.
lint:
	pylint -rn qiskit test tools
	tools/verify_headers.py qiskit test tools
	tools/find_optional_imports.py
	tools/find_stray_release_notes.py
	tools/verify_images.py

# Only pylint on files that have changed from origin/main. Also parallelize (disables cyclic-import check)
lint-incr:
	-git fetch -q https://github.com/Qiskit/qiskit-terra.git :lint_incr_latest
	tools/pylint_incr.py -j4 -rn -sn --paths :/qiskit/*.py :/test/*.py :/tools/*.py
	tools/verify_headers.py qiskit test tools
	tools/find_optional_imports.py
	tools/verify_images.py

ruff:
	ruff qiskit test tools setup.py

style:
	black --check qiskit test tools setup.py

black:
	black qiskit test tools setup.py

# Use the -s (starting directory) flag for "unittest discover" is necessary,
# otherwise the QuantumCircuit header will be modified during the discovery.
test:
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================
	python3 -m unittest discover -s test/python -t . -v
	@echo ================================================
	@echo Consider using tox as suggested in the CONTRIBUTING.MD guideline. For running the tests as the CI, use test_ci
	@echo ================================================

# Use pytest to run tests
pytest:
	pytest test/python

# Use pytest to run randomized tests
pytest_randomized:
	pytest test/randomized

test_ci:
	QISKIT_TEST_CAPTURE_STREAMS=1 stestr run

test_randomized:
	python3 -m unittest discover -s test/randomized -t . -v

coverage:
	coverage3 run --source qiskit -m unittest discover -s test/python -q
	coverage3 report

coverage_erase:
	coverage erase

clean: coverage_erase ;
