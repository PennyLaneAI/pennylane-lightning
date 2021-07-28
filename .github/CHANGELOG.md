# Release 0.17.0-dev

### New features

### Improvements

* Static versions of jQuery and Bootstrap are no longer included in the CSS theme. [(118)](https://github.com/PennyLaneAI/pennylane-lightning/pull/118)

* Add support for Controlled Phase Gate (CPhaseShift).
  [(#112)](https://github.com/PennyLaneAI/pennylane-lightning/issues/112)

* Move changelog to `.github` and add a changelog reminder.

### Breaking changes

### Bug fixes

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Thomas Bromley, Lee James O'Riordan

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
