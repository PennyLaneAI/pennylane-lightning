# Release 0.17.0-dev

### New features

* PennyLane-Lightning now supports all of the operations and observables of `default.qubit`.
  [#124](https://github.com/PennyLaneAI/pennylane-lightning/pull/124)

* C++ layer now supports float (32-bit) and double (64-bit) templated complex data.
  [(#113)](https://github.com/PennyLaneAI/pennylane-lightning/pull/113)

### Improvements

* PennyLane-Lightning can now be installed without compiling its C++ binaries and will fall back
  to using the `default.qubit` implementation. Skipping compilation is achieved by setting the
  `SKIP_COMPILATION` environment variable, e.g., Linux/MacOS: `export SKIP_COMPILATION=True`,
  Windows: `set SKIP_COMPILATION=True`. This feature is intended for building a pure-Python wheel of
  PennyLane-Lightning as a backup for platforms without a dedicated wheel.
  [(129)](https://github.com/PennyLaneAI/pennylane-lightning/pull/129)

* The C++-backed Python bound methods can now be directly called with wires and supplied parameters.
  [(125)](https://github.com/PennyLaneAI/pennylane-lightning/pull/125)

* Lightning supports arbitrary unitary and non-unitary gate-calls from Python to C++ layer.
  [(121)](https://github.com/PennyLaneAI/pennylane-lightning/pull/121)

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

* Column-major data in numpy is now correctly converted to row-major upon pass to the C++ layer.
  [(#126)](https://github.com/PennyLaneAI/pennylane-lightning/pull/126)

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
