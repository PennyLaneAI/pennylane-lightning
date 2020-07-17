lightning:
	g++ -I /home/tom/miniconda3/envs/pennylane-lightning/include/eigen3/ -O3 -shared -w -std=c++14 -fPIC `python3 -m pybind11 --includes` lightning_qubit_ops.cpp -o lightning_qubit_ops.so -DXTENSOR_USE_XSIMD -mavx2 -ffast-math -lblas -llapack

# Note: requires xtensor, xtensor-blas, xtensor-python, xsimd, pybind. Install using conda.
# conda install -c conda-forge xtensor
# conda install -c conda-forge xtensor-blas
# conda install -c conda-forge xtensor-python
# conda install -c conda-forge xsimd
# conda install -c conda-forge pybind11
# conda install -c conda-forge eigen