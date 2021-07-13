PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane_lightning --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=short

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
	rm -rf pennylane_lightning/__pycache__
	rm -rf pennylane_lightning/src/__pycache__
	rm -rf tests/__pycache__
	rm -rf pennylane_lightning/src/tests/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	rm -rf doc/code/api
	make -C doc clean

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)

test-cpp:
	make -C pennylane_lightning/src/tests clean
	GOOGLETEST_DIR=$(HOME)/googletest make -C pennylane_lightning/src/tests test

.PHONY: format
format:
ifdef check
	./bin/format --check pennylane_lightning/src tests
else
	./bin/format pennylane_lightning/src tests
endif
