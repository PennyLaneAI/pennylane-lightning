# Release 0.22.0-dev

### New features since last release

* Add Docker support.
[(#234)](https://github.com/PennyLaneAI/pennylane-lightning/pull/234)

### Breaking changes

### Improvements

* Update adjointJacobian and VJP methods.
[(#222)](https://github.com/PennyLaneAI/pennylane-lightning/pull/222)

* Set GitHub workflow to upload wheels to Test PyPI.
[(#220)](https://github.com/PennyLaneAI/pennylane-lightning/pull/220)

* Finalize the new kernel implementation.
[(#212)](https://github.com/PennyLaneAI/pennylane-lightning/pull/212)

### Documentation

### Bug fixes

* Fix a bug in Python tests with operations' `matrix` calls.
[(#238)](https://github.com/PennyLaneAI/pennylane-lightning/pull/238)

* Refactor utility header and fix a bug in linear algebra function with CBLAS.
[(#228)](https://github.com/PennyLaneAI/pennylane-lightning/pull/228)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Chae-Yeun Park

---

# Release 0.21.0

### New features since last release
* Direct support to probability, expectation value and variance calculation in PL-Lightning.
[(#185)](https://github.com/PennyLaneAI/pennylane-lightning/pull/185)

* Add C++ only benchmark for a given list of gates.
[(#199)](https://github.com/PennyLaneAI/pennylane-lightning/pull/199)

* Add bindings to C++ expval, var, probs.
[(#214)](https://github.com/PennyLaneAI/pennylane-lightning/pull/214)

### Breaking changes

### Improvements

* Add a new C++ kernel. 
[(#202)](https://github.com/PennyLaneAI/pennylane-lightning/pull/202)

* Ensure debug info is built into dynamic libraries.
[(#201)](https://github.com/PennyLaneAI/pennylane-lightning/pull/201)

### Documentation

### Bug fixes

* Add virtual destructor to C++ state-vector classes.
[(#200)](https://github.com/PennyLaneAI/pennylane-lightning/pull/200)

* Fix failed tests for the non-binary wheel.
[(#213)](https://github.com/PennyLaneAI/pennylane-lightning/pull/213)

* Fix failed tests on Windows.
[(#218)](https://github.com/PennyLaneAI/pennylane-lightning/pull/218)

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

* Add VJP support to PL-Lightning.
[(#181)](https://github.com/PennyLaneAI/pennylane-lightning/pull/181)

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
