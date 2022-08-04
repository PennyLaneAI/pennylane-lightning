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
	@echo "  docs               to generate documents"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  test-cpp           to run the C++ test suite"
	@echo "  test-python        to run the Python test suite"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to check or modify with clang-format-{version} instead of clang-format"
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
	$(MAKE) -C doc clean
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf dist
	rm -rf build
	rm -rf BuildTests BuildTidy BuildGBench
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat
	rm -rf pennylane_lightning/lightning_qubit_ops*

docs:
	$(MAKE) -C doc html

.PHONY : clean-docs
clean-docs:
	$(MAKE) -C doc clean

.PHONY : test-builtin test-suite test-python coverage coverage-cpp test-cpp test-cpp-no-omp test-cpp-blas test-cpp-kokkos
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

coverage-cpp:
	@echo "Generating cpp coverage report in BuildCov/out .."
	rm -rf ./BuildCov
	cmake pennylane_lightning/src -BBuildCov  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON
	cmake --build ./BuildCov
	cd ./BuildCov; ./tests/runner; \
	lcov --directory . -b ../pennylane_lightning/src --capture --output-file coverage.info; \
	genhtml coverage.info --output-directory out

test-cpp:
	rm -rf ./BuildTests
	cmake $(LIGHTNING_CPP_DIR) -BBuildTests -DBUILD_TESTS=ON
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

test-cpp-blas:
	rm -rf ./BuildTests
	cmake $(LIGHTNING_CPP_DIR) -BBuildTests -DBUILD_TESTS=ON -DENABLE_BLAS=ON
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

test-cpp-no-omp:
	rm -rf ./BuildTests
	cmake $(LIGHTNING_CPP_DIR) -BBuildTests -DBUILD_TESTS=ON -DENABLE_OPENMP=OFF
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

test-cpp-kokkos:
	rm -rf ./BuildTests
	cmake $(LIGHTNING_CPP_DIR) -BBuildTests -DBUILD_TESTS=ON -DENABLE_KOKKOS=ON
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

.PHONY: gbenchmark
gbenchmark:
	rm -rf ./BuildGBench
	cmake $(LIGHTNING_CPP_DIR) -BBuildGBench -DBUILD_BENCHMARKS=ON -DENABLE_OPENMP=ON -DENABLE_BLAS=ON -DCMAKE_BUILD_TYPE=Release -DBLA_VENDOR=OpenBLAS
	cmake --build ./BuildGBench 

.PHONY: format format-cpp format-python
format: format-cpp format-python

format-cpp:
ifdef check
	./bin/format --check --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning/src
else
	./bin/format --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning/src
endif

format-python:
ifdef check
	black -l 100 ./pennylane_lightning/ ./tests --check
else
	black -l 100 ./pennylane_lightning/ ./tests
endif

.PHONY: check-tidy
check-tidy:
	rm -rf ./BuildTidy
	cmake . -BBuildTidy -DENABLE_CLANG_TIDY=ON -DBUILD_TESTS=ON
	cmake --build ./BuildTidy
