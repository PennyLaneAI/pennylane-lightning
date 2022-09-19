# Release 0.26.0

### Improvements

* Introduces requirements-dev.txt and improves dockerfile.
[(#330)](https://github.com/PennyLaneAI/pennylane-lightning/pull/330)

* Support `expval` for a Hamiltonian.
[(#333)](https://github.com/PennyLaneAI/pennylane-lightning/pull/333)

* Implements caching for Kokkos installation.
[(#316)](https://github.com/PennyLaneAI/pennylane-lightning/pull/316)

* Supports measurements of operator arithmetic classes such as `Sum`, `Prod`,
  and `SProd` by deferring handling of them to `DefaultQubit`.
  [(#349)](https://github.com/PennyLaneAI/pennylane-lightning/pull/349)

```
@qml.qnode(qml.device('lightning.qubit', wires=2))
def circuit():
    obs = qml.s_prod(2.1, qml.PauliZ(0)) + qml.op_sum(qml.PauliX(0), qml.PauliZ(1))
    return qml.expval(obs)
```

### Bug fixes

* Test updates to reflect new measurement error messages.
[(#334)](https://github.com/PennyLaneAI/pennylane-lightning/pull/334)

* Updates to the release tagger to fix incompatibilities with RTD.
[(#344)](https://github.com/PennyLaneAI/pennylane-lightning/pull/344)

* Update cancel-workflow-action and bot credentials.
[(#345)](https://github.com/PennyLaneAI/pennylane-lightning/pull/345)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Christina Lee, Lee J. O'Riordan, Chae-Yeun Park

---

# Release 0.25.0

### New features since last release

### Breaking changes

* We explicitly disable support for PennyLane's parameter broadcasting.
[#317](https://github.com/PennyLaneAI/pennylane-lightning/pull/317)

* We explicitly remove support for PennyLane's `Sum`, `SProd` and `Prod`
  as observables.
  [(#326)](https://github.com/PennyLaneAI/pennylane-lightning/pull/326)

### Improvements

* CI builders use a reduced set of resources and redundant tests for PRs.
[(#319)](https://github.com/PennyLaneAI/pennylane-lightning/pull/319)

* Parallelize wheel-builds where applicable.
[(#314)](https://github.com/PennyLaneAI/pennylane-lightning/pull/314)

* AVX2/512 kernels are now available on Linux/MacOS with x86-64 architecture.
[(#313)](https://github.com/PennyLaneAI/pennylane-lightning/pull/313)

### Documentation

* Updated ReadTheDocs runner version from Ubuntu 20.04 to 22.04 
[(#327)](https://github.com/PennyLaneAI/pennylane-lightning/pull/327)

### Bug fixes

* Test updates to reflect new additions to PennyLane.
[(#318)](https://github.com/PennyLaneAI/pennylane-lightning/pull/318)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Christina Lee, Rashid N H M, Lee J. O'Riordan, Chae-Yeun Park

---

# Release 0.24.0

### New features since last release

* Add `SingleExcitation` and `DoubleExcitation` qchem gates and generators.
[(#289)](https://github.com/PennyLaneAI/pennylane-lightning/pull/289)

* Add a new dispatch mechanism for future kernels.
[(#291)](https://github.com/PennyLaneAI/pennylane-lightning/pull/291)

* Add `IsingXY` gate operation. 
[(#303)](https://github.com/PennyLaneAI/pennylane-lightning/pull/303)

* Support `qml.state()` in vjp and Hamiltonian in adjoint jacobian.
[(#294)](https://github.com/PennyLaneAI/pennylane-lightning/pull/294)

### Breaking changes

* Codebase is now moving to C++20. The default compiler for Linux is now GCC10.
[(#295)](https://github.com/PennyLaneAI/pennylane-lightning/pull/295)

* Minimum macOS version is changed to 10.15 (Catalina).
[(#295)](https://github.com/PennyLaneAI/pennylane-lightning/pull/295)

### Improvements

* Split matrix operations, refactor dispatch mechanisms, and add a benchmark suits.
[(#274)](https://github.com/PennyLaneAI/pennylane-lightning/pull/274)

* Add native support for the calculation of sparse Hamiltonians' expectation values. 
Sparse operations are offloaded to [Kokkos](https://github.com/kokkos/kokkos) and 
[Kokkos-Kernels](https://github.com/kokkos/kokkos-kernels).
[(#283)](https://github.com/PennyLaneAI/pennylane-lightning/pull/283)

* Device `lightning.qubit` now accepts a datatype for a statevector.
[(#290)](https://github.com/PennyLaneAI/pennylane-lightning/pull/290)

```python
dev1 = qml.device('lightning.qubit', wires=4, c_dtype=np.complex64) # for single precision
dev2 = qml.device('lightning.qubit', wires=4, c_dtype=np.complex128) # for double precision
```

### Documentation

* Use the centralized [Xanadu Sphinx Theme](https://github.com/XanaduAI/xanadu-sphinx-theme)
  to style the Sphinx documentation.
[(#287)](https://github.com/PennyLaneAI/pennylane-lightning/pull/287)

### Bug fixes

* Fix the issue with using available `clang-format` version in format.
[(#288)](https://github.com/PennyLaneAI/pennylane-lightning/pull/288)

* Fix a bug in the generator of `DoubleExcitationPlus`.
[(#298)](https://github.com/PennyLaneAI/pennylane-lightning/pull/298)

### Contributors

This release contains contributions from (in alphabetical order):

Mikhail Andrenkov, Ali Asadi, Amintor Dusko, Lee James O'Riordan, Chae-Yeun Park, and Shuli Shu

---

# Release 0.23.0

### New features since last release

* Add `generate_samples()` to lightning.
[(#247)](https://github.com/PennyLaneAI/pennylane-lightning/pull/247)

* Add Lightning GBenchmark Suite.
[(#249)](https://github.com/PennyLaneAI/pennylane-lightning/pull/249)

* Support runtime and compile information.
[(#253)](https://github.com/PennyLaneAI/pennylane-lightning/pull/253)

### Improvements

* Add `ENABLE_BLAS` build to CI checks.
[(#249)](https://github.com/PennyLaneAI/pennylane-lightning/pull/249)

* Add more `clang-tidy` checks and kernel tests.
[(#253)](https://github.com/PennyLaneAI/pennylane-lightning/pull/253)

* Add C++ code coverage to CI.
[(#265)](https://github.com/PennyLaneAI/pennylane-lightning/pull/265)

* Skip over identity operations in `"lightning.qubit"`.
[(#268)](https://github.com/PennyLaneAI/pennylane-lightning/pull/268)

### Bug fixes

* Update tests to remove `JacobianTape`.
[(#260)](https://github.com/PennyLaneAI/pennylane-lightning/pull/260)

* Fix tests for MSVC.
[(#264)](https://github.com/PennyLaneAI/pennylane-lightning/pull/264)

* Fix `#include <cpuid.h>` for PPC and AArch64 in Linux.
[(#266)](https://github.com/PennyLaneAI/pennylane-lightning/pull/266)

* Remove deprecated tape execution methods.
[(#270)](https://github.com/PennyLaneAI/pennylane-lightning/pull/270)

* Update `qml.probs` in `test_measures.py`.
[(#280)](https://github.com/PennyLaneAI/pennylane-lightning/pull/280)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Chae-Yeun Park, Lee James O'Riordan, and Trevor Vincent

---

# Release 0.22.1

### Bug fixes

* Ensure `Identity ` kernel is registered to C++ dispatcher.
[(#275)](https://github.com/PennyLaneAI/pennylane-lightning/pull/275)

---

# Release 0.22.0

### New features since last release

* Add Docker support.
[(#234)](https://github.com/PennyLaneAI/pennylane-lightning/pull/234)

### Improvements

* Update quantum tapes serialization and Python tests.
[(#239)](https://github.com/PennyLaneAI/pennylane-lightning/pull/239)

* Clang-tidy is now enabled for both tests and examples builds under Github Actions.
[(#237)](https://github.com/PennyLaneAI/pennylane-lightning/pull/237)

* The return type of `StateVectorBase` data is now derived-class defined.
[(#237)](https://github.com/PennyLaneAI/pennylane-lightning/pull/237)

* Update adjointJacobian and VJP methods.
[(#222)](https://github.com/PennyLaneAI/pennylane-lightning/pull/222)

* Set GitHub workflow to upload wheels to Test PyPI.
[(#220)](https://github.com/PennyLaneAI/pennylane-lightning/pull/220)

* Finalize the new kernel implementation.
[(#212)](https://github.com/PennyLaneAI/pennylane-lightning/pull/212)

### Documentation

* Use of batching with OpenMP threads is documented.
[(#221)](https://github.com/PennyLaneAI/pennylane-lightning/pull/221)

### Bug fixes

* Fix for OOM errors when using adjoint with large numbers of observables.
[(#221)](https://github.com/PennyLaneAI/pennylane-lightning/pull/221)

* Add virtual destructor to C++ state-vector classes.
[(#200)](https://github.com/PennyLaneAI/pennylane-lightning/pull/200)

* Fix a bug in Python tests with operations' `matrix` calls.
[(#238)](https://github.com/PennyLaneAI/pennylane-lightning/pull/238)

* Refactor utility header and fix a bug in linear algebra function with CBLAS.
[(#228)](https://github.com/PennyLaneAI/pennylane-lightning/pull/228)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Chae-Yeun Park, Lee James O'Riordan

---

# Release 0.21.0

### New features since last release

* Add C++ only benchmark for a given list of gates.
[(#199)](https://github.com/PennyLaneAI/pennylane-lightning/pull/199)

* Wheel-build support for Python 3.10.
[(#186)](https://github.com/PennyLaneAI/pennylane-lightning/pull/186)

* C++ support for probability, expectation value and variance calculations.
[(#185)](https://github.com/PennyLaneAI/pennylane-lightning/pull/185)

* Add bindings to C++ expval, var, probs.
[(#214)](https://github.com/PennyLaneAI/pennylane-lightning/pull/214)

### Improvements

* `setup.py` adds debug only when --debug is given
[(#208)](https://github.com/PennyLaneAI/pennylane-lightning/pull/208)

* Add new highly-performant C++ kernels for quantum gates. 
[(#202)](https://github.com/PennyLaneAI/pennylane-lightning/pull/202)

The new kernels significantly improve the runtime performance of PennyLane-Lightning 
for both differentiable and non-differentiable workflows. Here is an example workflow
using the adjoint differentiation method with a circuit of 5 strongly entangling layers:

```python
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from numpy.random import random
np.random.seed(42)
n_layers = 5
n_wires = 6
dev = qml.device("lightning.qubit", wires=n_wires)

@qml.qnode(dev, diff_method="adjoint")
def circuit(weights):
    StronglyEntanglingLayers(weights, wires=list(range(n_wires)))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

init_weights = np.random.random(StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires))
params = np.array(init_weights,requires_grad=True)
jac = qml.jacobian(circuit)(params)
```
The latest release shows improved performance on both single and multi-threaded evaluations!

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane-lightning/v0.21.0-rc0/doc/_static/lightning_v20_v21_bm.png" width=50%/>

* Ensure debug info is built into dynamic libraries.
[(#201)](https://github.com/PennyLaneAI/pennylane-lightning/pull/201)

### Documentation

* New guidelines on adding and benchmarking C++ kernels.
[(#202)](https://github.com/PennyLaneAI/pennylane-lightning/pull/202)

### Bug fixes

* Update clang-format version
[(#219)](https://github.com/PennyLaneAI/pennylane-lightning/pull/219)

* Fix failed tests on Windows.
[(#218)](https://github.com/PennyLaneAI/pennylane-lightning/pull/218)

* Update clang-format version
[(#219)](https://github.com/PennyLaneAI/pennylane-lightning/pull/219)

* Add virtual destructor to C++ state-vector classes.
[(#200)](https://github.com/PennyLaneAI/pennylane-lightning/pull/200)

* Fix failed tests for the non-binary wheel.
[(#213)](https://github.com/PennyLaneAI/pennylane-lightning/pull/213)

* Add virtual destructor to C++ state-vector classes.
[(#200)](https://github.com/PennyLaneAI/pennylane-lightning/pull/200)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Chae-Yeun Park, Lee James O'Riordan

---

# Release 0.20.1

### Bug fixes

* Fix missing header-files causing build errors in algorithms module.
[(#193)](https://github.com/PennyLaneAI/pennylane-lightning/pull/193)

* Fix failed tests for the non-binary wheel.
[(#191)](https://github.com/PennyLaneAI/pennylane-lightning/pull/191)

---
# Release 0.20.2

### Bug fixes

* Introduce CY kernel to Lightning to avoid issues with decomposition.
[(#203)](https://github.com/PennyLaneAI/pennylane-lightning/pull/203)

### Contributors

This release contains contributions from (in alphabetical order):

Lee J. O'Riordan

# Release 0.20.1

### Bug fixes

* Fix missing header-files causing build errors in algorithms module.
[(#193)](https://github.com/PennyLaneAI/pennylane-lightning/pull/193)

* Fix failed tests for the non-binary wheel.
[(#191)](https://github.com/PennyLaneAI/pennylane-lightning/pull/191)

# Release 0.20.0

### New features since last release

* Add wheel-builder support for Python 3.10.
  [(#186)](https://github.com/PennyLaneAI/pennylane-lightning/pull/186)

* Add VJP support to PL-Lightning.
[(#181)](https://github.com/PennyLaneAI/pennylane-lightning/pull/181)

* Add complex64 support in PL-Lightning.
[(#177)](https://github.com/PennyLaneAI/pennylane-lightning/pull/177)

* Added examples folder containing aggregate gate performance test.
[(#165)](https://github.com/PennyLaneAI/pennylane-lightning/pull/165)

### Breaking changes

### Improvements

* Update PL-Lightning to support new features in PL.
[(#179)](https://github.com/PennyLaneAI/pennylane-lightning/pull/179)

### Documentation

* Lightning setup.py build process uses CMake.
[(#176)](https://github.com/PennyLaneAI/pennylane-lightning/pull/176)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Chae-Yeun Park, Isidor Schoch, Lee James O'Riordan

---

# Release 0.19.0

* Add Cache-Friendly DOTC, GEMV, GEMM along with BLAS Support.
[(#155)](https://github.com/PennyLaneAI/pennylane-lightning/pull/155)

### Improvements

* The performance of parametric gates has been improved.
  [(#157)](https://github.com/PennyLaneAI/pennylane-lightning/pull/157)

* AVX support is enabled for Linux users on Intel/AMD platforms.
  [(#157)](https://github.com/PennyLaneAI/pennylane-lightning/pull/157)

* PennyLane-Lightning has been updated to conform with clang-tidy
  recommendations for modernization, offering performance improvements across
  all use-cases.
  [(#153)](https://github.com/PennyLaneAI/pennylane-lightning/pull/153)

### Breaking changes

* Linux users on `x86_64` must have a CPU supporting AVX.
  [(#157)](https://github.com/PennyLaneAI/pennylane-lightning/pull/157)

### Bug fixes

* OpenMP built with Intel MacOS CI runners causes failures on M1 Macs. OpenMP is currently
  disabled in the built wheels until this can be resolved with Github Actions runners.
  [(#166)](https://github.com/PennyLaneAI/pennylane-lightning/pull/166)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Lee James O'Riordan

---

# Release 0.18.0

### New features since last release

* PennyLane-Lightning now provides a high-performance
  [adjoint Jacobian](http://arxiv.org/abs/2009.02823) method for differentiating quantum circuits.
  [(#136)](https://github.com/PennyLaneAI/pennylane-lightning/pull/136)
  
  The adjoint method operates after a forward pass by iteratively applying inverse gates to scan
  backwards through the circuit. The method is already available in PennyLane's
  `default.qubit` device, but the version provided by `lightning.qubit` integrates with the C++
  backend and is more performant, as shown in the plot below:

  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane-lightning/master/doc/_static/lightning_adjoint.png" width=70%/>
  
  The plot compares the average runtime of `lightning.qubit` and `default.qubit` for calculating the
  Jacobian of a circuit using the adjoint method for a range of qubit numbers. The circuit
  consists of ten `BasicEntanglerLayers` with a `PauliZ` expectation value calculated on each wire,
  repeated over ten runs. We see that `lightning.qubit` provides a speedup of around two to eight
  times, depending on the number of qubits.
  
  The adjoint method can be accessed using the standard interface. Consider the following circuit:
  
  ```python
  import pennylane as qml

  wires = 3
  layers = 2
  dev = qml.device("lightning.qubit", wires=wires)

  @qml.qnode(dev, diff_method="adjoint")
  def circuit(weights):
      qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
      return qml.expval(qml.PauliZ(0))

  weights = qml.init.strong_ent_layers_normal(layers, wires, seed=1967)
  ```
  
  The circuit can be executed and its gradient calculated using:
  
    ```pycon
  >>> print(f"Circuit evaluated: {circuit(weights)}")
  Circuit evaluated: 0.9801286266677633
  >>> print(f"Circuit gradient:\n{qml.grad(circuit)(weights)}")
  Circuit gradient:
  [[[-1.11022302e-16 -1.63051504e-01 -4.14810501e-04]
    [ 1.11022302e-16 -1.50136528e-04 -1.77922957e-04]
    [ 0.00000000e+00 -3.92874550e-02  8.14523075e-05]]

   [[-1.14472273e-04  3.85963953e-02  0.00000000e+00]
    [-5.76791765e-05 -9.78478343e-02  0.00000000e+00]
    [-5.55111512e-17  0.00000000e+00 -1.11022302e-16]]]  
  ```

* PennyLane-Lightning now supports all of the operations and observables of `default.qubit`.
  [(#124)](https://github.com/PennyLaneAI/pennylane-lightning/pull/124)

### Improvements

* A new state-vector class `StateVectorManaged` was added, enabling memory use to be bound to 
  statevector lifetime.
  [(#136)](https://github.com/PennyLaneAI/pennylane-lightning/pull/136)

* The repository now has a well-defined component hierarchy, allowing each indepedent unit to be 
  compiled and linked separately.
  [(#136)](https://github.com/PennyLaneAI/pennylane-lightning/pull/136)

* PennyLane-Lightning can now be installed without compiling its C++ binaries and will fall back
  to using the `default.qubit` implementation. Skipping compilation is achieved by setting the
  `SKIP_COMPILATION` environment variable, e.g., Linux/MacOS: `export SKIP_COMPILATION=True`,
  Windows: `set SKIP_COMPILATION=True`. This feature is intended for building a pure-Python wheel of
  PennyLane-Lightning as a backup for platforms without a dedicated wheel.
  [(#129)](https://github.com/PennyLaneAI/pennylane-lightning/pull/129)

* The C++-backed Python bound methods can now be directly called with wires and supplied parameters.
  [(#125)](https://github.com/PennyLaneAI/pennylane-lightning/pull/125)

* Lightning supports arbitrary unitary and non-unitary gate-calls from Python to C++ layer.
  [(#121)](https://github.com/PennyLaneAI/pennylane-lightning/pull/121)

### Documentation

* Added preliminary architecture diagram for package.
  [(#131)](https://github.com/PennyLaneAI/pennylane-lightning/pull/131)

* C++ API built as part of docs generation.
  [(#131)](https://github.com/PennyLaneAI/pennylane-lightning/pull/131)

### Breaking changes

* Wheels for MacOS <= 10.13 will no longer be provided due to XCode SDK C++17 support requirements.
  [(#149)](https://github.com/PennyLaneAI/pennylane-lightning/pull/149)

### Bug fixes

* An indexing error in the CRY gate is fixed. [(#136)](https://github.com/PennyLaneAI/pennylane-lightning/pull/136)

* Column-major data in numpy is now correctly converted to row-major upon pass to the C++ layer.
  [(#126)](https://github.com/PennyLaneAI/pennylane-lightning/pull/126)

### Contributors

This release contains contributions from (in alphabetical order):

Thomas Bromley, Lee James O'Riordan

---

# Release 0.17.0

### New features

* C++ layer now supports float (32-bit) and double (64-bit) templated complex data.
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)

### Improvements

* The PennyLane device test suite is now included in coverage reports.
  [(#123)](https://github.com/PennyLaneAI/pennylane-lightning/pull/123)

* Static versions of jQuery and Bootstrap are no longer included in the CSS theme. 
  [(#118)](https://github.com/PennyLaneAI/pennylane-lightning/pull/118)

* C++ tests have been ported to use Catch2 framework.
  [(#115)](https://github.com/PennyLaneAI/pennylane-lightning/pull/115)

* Testing now exists for both float and double precision methods in C++ layer. 
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)
  [(#115)](https://github.com/PennyLaneAI/pennylane-lightning/pull/115)

* Compile-time utility methods with `constexpr` have been added.
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)

* Wheel-build support for ARM64 (Linux and MacOS) and PowerPC (Linux) added. 
  [(#110)](https://github.com/PennyLaneAI/pennylane-lightning/pull/110)

* Add support for Controlled Phase Gate (`ControlledPhaseShift`).
  [(#114)](https://github.com/PennyLaneAI/pennylane-lightning/pull/114)

* Move changelog to `.github` and add a changelog reminder.
  [(#111)](https://github.com/PennyLaneAI/pennylane-lightning/pull/111)

* Adds CMake build system support.
  [(#104)](https://github.com/PennyLaneAI/pennylane-lightning/pull/104)


### Breaking changes

* Removes support for Python 3.6 and adds support for Python 3.9.
  [(#127)](https://github.com/PennyLaneAI/pennylane-lightning/pull/127)
  [(#128)](https://github.com/PennyLaneAI/pennylane-lightning/pull/128)

* Compilers with C++17 support are now required to build C++ module.
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)

* Gate classes have been removed with functionality added to StateVector class.
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)

* We are no longer building wheels for Python 3.6.
  [(#106)](https://github.com/PennyLaneAI/pennylane-lightning/pull/106)

### Bug fixes

* PowerPC wheel-builder now successfully compiles modules.
  [(#120)](https://github.com/PennyLaneAI/pennylane-lightning/pull/120)

### Documentation

* Added community guidelines.
  [(#109)](https://github.com/PennyLaneAI/pennylane-lightning/pull/109)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Christina Lee, Thomas Bromley, Lee James O'Riordan

---

# Release 0.15.1

### Bug fixes

* The PennyLane-Lightning binaries are now built with NumPy 1.19.5, to avoid ABI
  compatibility issues with the latest NumPy 1.20 release. See
  [the NumPy release notes](https://numpy.org/doc/stable/release/1.20.0-notes.html#size-of-np-ndarray-and-np-void-changed)
  for more details.
  [(#97)](https://github.com/PennyLaneAI/pennylane-lightning/pull/97)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Antal Száva

---

# Release 0.15.0

### Improvements

* For compatibility with PennyLane v0.15, the `analytic` keyword argument
  has been removed. Statistics can still be computed analytically by setting
  `shots=None`.
  [(#93)](https://github.com/PennyLaneAI/pennylane-lightning/pull/93)

* Inverse gates are now supported.
  [(#89)](https://github.com/PennyLaneAI/pennylane-lightning/pull/89)

* Add new lightweight backend with performance improvements.
  [(#57)](https://github.com/PennyLaneAI/pennylane-lightning/pull/57)

* Remove the previous Eigen-based backend.
  [(#67)](https://github.com/PennyLaneAI/pennylane-lightning/pull/67)

### Bug fixes

* Re-add dispatch table after fixing static initialisation order issue.
  [(#68)](https://github.com/PennyLaneAI/pennylane-lightning/pull/68)

### Contributors

This release contains contributions from (in alphabetical order):

Thomas Bromley, Theodor Isacsson, Christina Lee, Thomas Loke, Antal Száva.

---

# Release 0.14.1

### Bug fixes

* Fixes a bug where the `QNode` would swap `LightningQubit` to
  `DefaultQubitAutograd` on device execution due to the inherited
  `passthru_devices` entry of the `capabilities` dictionary.
  [(#61)](https://github.com/PennyLaneAI/pennylane-lightning/pull/61)

### Contributors

This release contains contributions from (in alphabetical order):

Antal Száva

---

# Release 0.14.0

### Improvements

* Extends support from 16 qubits to 50 qubits.
  [(#52)](https://github.com/PennyLaneAI/pennylane-lightning/pull/52)

### Bug fixes

* Updates applying basis state preparations to correspond to the
  changes in `DefaultQubit`.
  [(#55)](https://github.com/PennyLaneAI/pennylane-lightning/pull/55)

### Contributors

This release contains contributions from (in alphabetical order):

Thomas Loke, Tom Bromley, Josh Izaac, Antal Száva

---

# Release 0.12.0

### Bug fixes

* Updates capabilities dictionary to be compatible with core PennyLane
  [(#45)](https://github.com/PennyLaneAI/pennylane-lightning/pull/45)

* Fix install of Eigen for CI wheel building
  [(#44)](https://github.com/PennyLaneAI/pennylane-lightning/pull/44)

### Contributors

This release contains contributions from (in alphabetical order):

Tom Bromley, Josh Izaac, Antal Száva

---

# Release 0.11.0

Initial release.

This release contains contributions from (in alphabetical order):

Tom Bromley, Josh Izaac, Nathan Killoran, Antal Száva
