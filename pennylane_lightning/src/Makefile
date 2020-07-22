# Compile with minimum number of compiler flags, hopefully taking the least amount of time.
dev:
	g++ -I /usr/include/eigen3 -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` lightning_qubit.cpp -o lightning_qubit_ops.so

# Extra compiler flags that I don't understand. We should think about which to include.
lightning:
	g++ -I /usr/include/eigen3 -O3 -shared -w -std=c++11 -fPIC `python3 -m pybind11 --includes` lightning_qubit.cpp -o lightning_qubit_ops.so -DXTENSOR_USE_XSIMD -mavx2 -ffast-math -lblas -llapack