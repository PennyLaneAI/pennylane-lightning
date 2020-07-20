dev:
	g++ -I /usr/include/eigen3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` lightning_qubit_ops.cpp -o lightning_qubit_ops.so

lightning:
	g++ -I /usr/include/eigen3 -O3 -shared -w -std=c++14 -fPIC `python3 -m pybind11 --includes` lightning_qubit_ops.cpp -o lightning_qubit_ops.so -DXTENSOR_USE_XSIMD -mavx2 -ffast-math -lblas -llapack