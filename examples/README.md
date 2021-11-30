# Gate aggregate performance tests
Run `bash run_gate_benchmark.sh $CXX_COMPILER`, where `$CXX_COMPILER` is the compiler you wish to use, in the terminal (e.g. `bash run_gate_benchmark.sh clang++`). The script will automatically build the gate_benchmark project.
It will set the CXX environment variable to "$CXX_COMPILER".

## Implementation details: 
* The compile-time options are controlled by the bash script `run_gate_benchmark.sh`
* The PennyLane-Lightning benchmark is provided in the `gate_benchmark.cpp` file
* Plotting is accomplished with the Python script `gate_benchmark_plotter.py`. 
* Plotting requires the packages listed in `requirements.txt`
* The number of gate repetitions is set to 3 and can be changed in the bash script `run_gate_benchmark.sh` by modifying the `num_gate_reps` variable

### `gate_benchmark.cpp`:
* A single random angle is generated per gate repetition and qubit; the same random angle is used once for all of the parameterised gates
* The gates are applied in the order X, Y, Z, H, CNOT, CZ, RX, RY, RZ, CRX, CRY, CRZ
* The above order is repeated `num_gate_reps`-times

### `gate_benchmark_plotter.py`:
* The first plot shows the absolute runtime
* The second plot is on a loglog scale which better depicts the exponential scaling of the relative runtime with respect to the number of simulated qubits
* We plot the time needed to execute the gate sequence averaged over the repetitions