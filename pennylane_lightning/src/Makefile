.PHONY: help clean test format

help:
	@echo "Please use \`make <target>\` where <target> is one of"
	@echo "  clean"
	@echo "  format"

clean:
	rm -rf ./build/* BuildTests BuildBench

test:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DBUILD_TESTS=ON
	cmake --build ./BuildTests --target runner
	cmake --build ./BuildTests --target test

benchmark:
	cmake --build BuildBench --target clean || true
	rm -rf ./BBuildBench/CMakeCache.txt
	cmake . -BBuildBench -DBUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX=ON -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${PWD}/BuildBench"
	cmake --build ./BuildBench
