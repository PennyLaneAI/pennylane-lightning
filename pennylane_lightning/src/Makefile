.PHONY: help clean format

help:
	@echo "Please use \`make <target>\` where <target> is one of"
	@echo "  clean"
	@echo "  format"

clean:
	rm -rf ./build/* 

test:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DBUILD_TESTS=1
	cmake --build ./BuildTests

