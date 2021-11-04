Run "bash run_gate_benchmark.sh" in the terminal. The script will automatically build the gate_benchmark project.
It will set the CXX and CMAKE_CXX_FLAGS environment variables to "which clang++" and "-O3".

Some comments on the implementation:
General comments: 
* I decided to only run the lightning code in the cpp file and to control the compilation, file-creation etc via a bash script. Plotting is done with the help of a python script as this offers the most code flexbility.
* I passed the compiler flags via the bash script because this allows me to automate the inclusion of these flags in the plots
* I assume clang is installed
* I build the project in the bash script to ensure that the compiler and flags are consistent with the ones that appear in the plot description
* The number of gate repitions was set to 3 as described in the email

gate_benchmark.cpp:
* I decided to generate a single random angle per parameterised gate as I feel like this makes use of parametric property of the gates while keeping the performance penalty due to cache access to a bare minimum
* I decided to apply the gates in the order that they were listed in the task; furthermore I perform the gate repitition after every single gate has been applied to avoid performing identity operations for gates which are involutory
* I have included many different libraries without much regard as to which libraries are already used in pennylane-lightning as I don't believe this is the focus of this task

gate_benchmark_plotter.py:
* The first plot is supposed to be close to the task description
* Due to the exponential nature of quantum computing one cannot read out the time for few qubits; I therefore also included a loglog plot depicting the scaling of the program
* Only the average is plotted as described in the task description. Normally I would average over many more runs and plot the 95% confidence interval or standard deviation