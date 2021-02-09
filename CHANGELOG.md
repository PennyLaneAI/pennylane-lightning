# Release 0.15.0-dev

### New features since last release

### Breaking changes

### Improvements

### Documentation

### Bug fixes

* Fixes a bug where the `QNode` would swap `LightningQubit` to
  `DefaultQubitAutograd` on device execution due to the inherited
  `passthru_devices` entry of the `capabilities` dictionary.
  [(#61)](https://github.com/PennyLaneAI/pennylane-lightning/pull/61)

### Contributors

This release contains contributions from (in alphabetical order):

Josh Izaac, Antal Száva

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
