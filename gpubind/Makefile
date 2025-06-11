PYTHON := python
PKG_NAME := gpu_binding_example

.PHONY: all install editable run clean help

# Default target
all: install

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Builds and installs the package into your environment."
	@echo "  editable      Installs in 'editable' mode for development."
	@echo "  run           Runs the main Python script using the installed package."
	@echo "  clean         Removes build artifacts and uninstalls the package."

install:
	$(PYTHON) -m pip install -e .

# Removes all generated build artifacts and uninstalls the package
clean:
	$(PYTHON) -m pip uninstall -y $(PKG_NAME)
	rm -rf build dist .eggs *.egg-info