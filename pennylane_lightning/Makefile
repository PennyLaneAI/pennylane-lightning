lightning:
	g++ -O3 -shared -w -std=c++14 -fPIC `python3 -m pybind11 --includes` lightning_qubit_ops.cpp -o lightning_qubit_ops.so -DXTENSOR_USE_XSIMD -mavx2 -ffast-math -lblas -llapack

# Note: requires xtensor, xtensor-blas, xtensor-python, xsimd, pybind. Install using conda.
