.PHONY: help clean test format

help:
	@echo "Please use \`make <target>\` where <target> is one of"
	@echo "  clean"
	@echo "  format"

clean:
	rm -rf ./build/* BuildTests

test:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DBUILD_TESTS=1
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test
