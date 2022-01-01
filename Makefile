PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane_lightning --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=short

LIGHTNING_CPP_DIR := pennylane_lightning/src/

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install PennyLane-Lightning"
	@echo "  wheel              to build the PennyLane-Lightning wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  test-cpp           to run the C++ test suite"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)"
	@echo "  check-tidy         to build PennyLane-Lightning with ENABLE_CLANG_TIDY=ON (requires clang-tidy & CMake)"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install PennyLane-Lightning you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	$(PYTHON) setup.py clean --all
	$(MAKE) -C $(LIGHTNING_CPP_DIR) clean
	$(MAKE) -C doc clean
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf dist
	rm -rf build
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat
	rm -rf pennylane_lightning/lightning_qubit_ops*

docs:
	$(MAKE) -C doc html

.PHONY : clean-docs
clean-docs:
	$(MAKE) -C doc clean

test-builtin:
	$(PYTHON) -I $(TESTRUNNER)

test-suite:
	pl-device-test --device lightning.qubit --skip-ops --shots=20000
	pl-device-test --device lightning.qubit --shots=None --skip-ops

test-python: test-builtin test-suite

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
	pl-device-test --device lightning.qubit --skip-ops --shots=20000 $(COVERAGE) --cov-append
	pl-device-test --device lightning.qubit --shots=None --skip-ops $(COVERAGE) --cov-append

test-cpp:
	$(MAKE) -C $(LIGHTNING_CPP_DIR) test

.PHONY: format format-cpp format-python
format: format-cpp format-python

format-cpp:
ifdef check
	./bin/format --check
else
	./bin/format
endif

format-python:
ifdef check
	black -l 100 pennylane_lightning/ --check
else
	black -l 100 pennylane_lightning/
endif

.PHONY: check-tidy
check-tidy:
	rm -rf ./Build
	cmake . -BBuild -DENABLE_CLANG_TIDY=ON -DBUILD_TESTS=1
	cmake --build ./Build
