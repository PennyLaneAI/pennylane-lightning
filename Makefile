PYTHON := python3
COVERAGE := --cov=pennylane_lightning --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=short

PL_BACKEND ?= "$(if $(backend:-=),$(backend),lightning_qubit)"

ifdef check
    CHECK := --check --diff
    ICHECK := --check
else
    CHECK :=
    ICHECK :=
endif

ifdef build_options
    OPTIONS := $(build_options)
else
    OPTIONS :=
endif

ifdef verbose
    VERBOSE := --verbose
else
    VERBOSE :=
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  docs                     to generate documents"
	@echo "  clean                    to delete all temporary, cache, and build files"
	@echo "  clean-docs               to delete all built documentation"
	@echo "  test-cpp [backend=?]     to run the C++ test suite (requires CMake)"
	@echo "                           Default: lightning_qubit"
	@echo "  test-cpp [verbose=1]     to run the C++ test suite (requires CMake)"
	@echo "                           use with 'verbose=1' for building with verbose flag"
	@echo "  test-cpp [target=?]      to run a specific C++ test target (requires CMake)."
	@echo "  test-python [device=?]   to run the Python test suite"
	@echo "                           Default: lightning.qubit"
	@echo "  coverage [device=?]      to generate a coverage report for python interface"
	@echo "                           Default: lightning.qubit"
	@echo "  coverage-cpp [backend=?] to generate a coverage report for C++ interface"
	@echo "                           Default: lightning_qubit"
	@echo "  format [check=1]         to apply C++ and Python formatter;"
	@echo "                           use with 'check=1' to check instead of modify (requires black and clang-format)"
	@echo "  check-tidy [backend=?]   to build PennyLane-Lightning with ENABLE_CLANG_TIDY=ON (requires clang-tidy & CMake)"
	@echo "                           Default: lightning_qubit"
	@echo "  check-tidy [verbose=1]   to build PennyLane-Lightning with ENABLE_CLANG_TIDY=ON (requires clang-tidy & CMake)"
	@echo "                           use with 'verbose=1' for building with verbose flag"
	@echo "  check-tidy [target=?]    to build a specific PennyLane-Lightning target with ENABLE_CLANG_TIDY=ON (requires clang-tidy & CMake)"
	@echo "  docker-build [target=?]  to build a Docker image for a specific PennyLane-Lightning target"
	@echo "  docker-push  [target=?]  to push a Docker image to the PennyLaneAI Docker Hub repo"
	@echo "  docker-all  		      to build and push Docker images for all PennyLane-Lightning targets"

.PHONY: clean
clean:
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf build Build BuildTests BuildTidy
	rm -rf build_*
	rm -rf .coverage coverage_html_report/
	rm -rf pennylane_lightning/*_ops*
	rm -rf *.egg-info

.PHONY: coverage coverage-cpp
coverage:
	@echo "Generating coverage report for $(if $(device:-=),$(device),lightning.qubit) device:"
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
	pl-device-test --device $(if $(device:-=),$(device),lightning.qubit) --skip-ops --shots=20000 $(COVERAGE) --cov-append
	pl-device-test --device $(if $(device:-=),$(device),lightning.qubit) --shots=None --skip-ops $(COVERAGE) --cov-append

coverage-cpp:
	@echo "Generating cpp coverage report in BuildCov/out for $(PL_BACKEND) backend"
	rm -rf ./BuildCov
	cmake -BBuildCov -G Ninja \
		  -DCMAKE_BUILD_TYPE=Debug \
		  -DBUILD_TESTS=ON \
		  -DENABLE_COVERAGE=ON \
		  -DPL_BACKEND=$(PL_BACKEND) \
		  $(OPTIONS)
	cmake --build ./BuildCov
	cd ./BuildCov; for file in *runner ; do ./$file; done; \
	lcov --directory . -b ../pennylane_lightning/core/src --capture --output-file coverage.info; \
	genhtml coverage.info --output-directory out

.PHONY: test-python test-builtin test-suite test-cpp
test-python: test-builtin test-suite

test-builtin:
	PL_DEVICE=$(if $(device:-=),$(device),lightning.qubit) $(PYTHON) -I $(TESTRUNNER)

test-suite:
	pl-device-test --device $(if $(device:-=),$(device),lightning.qubit) --skip-ops --shots=20000
	pl-device-test --device $(if $(device:-=),$(device),lightning.qubit) --shots=None --skip-ops

test-cpp:
	rm -rf ./BuildTests
	cmake -BBuildTests -G Ninja \
		  -DCMAKE_BUILD_TYPE=Debug \
		  -DBUILD_TESTS=ON \
		  -DENABLE_WARNINGS=ON \
		  -DPL_BACKEND=$(PL_BACKEND) \
		  $(OPTIONS)
ifdef target
	cmake --build ./BuildTests $(VERBOSE) --target $(target)
	./BuildTests/$(target)
else
	cmake --build ./BuildTests $(VERBOSE)
	cmake --build ./BuildTests $(VERBOSE) --target test
endif

.PHONY: format format-cpp format-python
format: format-cpp format-python

format-cpp:
	./bin/format $(CHECK) ./pennylane_lightning

format-python:
	isort --py 311 --profile black -l 100 -p pennylane_lightning ./pennylane_lightning/ ./mpitests ./tests $(ICHECK) $(VERBOSE)
	black -l 100 ./pennylane_lightning/ ./mpitests ./tests $(CHECK) $(VERBOSE)

.PHONY: check-tidy
check-tidy:
	rm -rf ./BuildTidy
	cmake -BBuildTidy -G Ninja \
		  -DENABLE_CLANG_TIDY=ON \
		  -DBUILD_TESTS=ON \
		  -DENABLE_WARNINGS=ON \
		  -DCLANG_TIDY_BINARY=clang-tidy \
		  -DPL_BACKEND=$(PL_BACKEND) \
		  $(OPTIONS)
ifdef target
	cmake --build ./BuildTidy $(VERBOSE) --target $(target)
else
	cmake --build ./BuildTidy $(VERBOSE)
endif

.PHONY : docs clean-docs
docs:
	$(MAKE) -C doc html

clean-docs:
	$(MAKE) -C doc clean

.PHONY : docker-build docker-push docker-all
ifdef target
    TARGET := $(target)
else
    TARGET := lightning-qubit
endif
ifdef version
    VERSION := $(version)
else
    VERSION := v0.36.0
endif

docker-build:
	docker build -f docker/Dockerfile \
		  --tag=pennylaneai/pennylane:$(VERSION)-$(TARGET) \
		  --target wheel-$(TARGET) \
		  --build-arg='LIGHTNING_VERSION=$(VERSION)' .
docker-push:
	docker push pennylaneai/pennylane:$(VERSION)-$(TARGET)
docker-build-all:
	$(MAKE) docker-build target=lightning-qubit
	$(MAKE) docker-build target=lightning-gpu
	$(MAKE) docker-build target=lightning-kokkos-openmp
	$(MAKE) docker-build target=lightning-kokkos-cuda
	$(MAKE) docker-build target=lightning-kokkos-rocm
docker-push-all:
	$(MAKE) docker-push target=lightning-qubit
	$(MAKE) docker-push target=lightning-gpu
	$(MAKE) docker-push target=lightning-kokkos-openmp
	$(MAKE) docker-push target=lightning-kokkos-cuda
	$(MAKE) docker-push target=lightning-kokkos-rocm
docker-all:
	$(MAKE) docker-build-all
	$(MAKE) docker-push-all
