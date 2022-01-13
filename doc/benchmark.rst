Benchmark different kernel implementations
##########################################

You can benchmark different gate implementations inside the project.

.. code-block:: console

    $ make benchmark
    $ cd BuildBench

Inside the directory, you can see ``run_gate_benchmark.sh``. You can benchmark a specific kernel implementation for a gate by providing arguments. 
For example, you can compare benchmark results for ``PauliX`` gate with ``KernelType::PI`` and ``KernelType::LM`` by running the following commands:

.. code-block:: console
   
    $ ./run_gate_benchmark.sh PI PauliX
    $ ./run_gate_benchmark.sh LM PauliX

The benchmark results will be written in ``res_{COMPILER}_{VERSION}`` subdirectory. For example, if you use GCC version 9.3.0, the directory name will be ``res_GNU_9.3.0``. 

.. Add an instruction to plot
