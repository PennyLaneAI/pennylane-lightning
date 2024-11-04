# Release 0.39.0

### New features since last release

* Add support for out-of-order `qml.probs` in `lightning.gpu`.
  [(#941)](https://github.com/PennyLaneAI/pennylane-lightning/pull/941)

* Add mid-circuit measurements support to `lightning.gpu`'s single-GPU backend.
  [(#931)](https://github.com/PennyLaneAI/pennylane-lightning/pull/931)

* Integrate Lightning-GPU with Catalyst so that hybrid programs can be seamlessly QJIT-compiled and executed on this device following `pip install pennylane-lightning-gpu`.
  [(#928)](https://github.com/PennyLaneAI/pennylane-lightning/pull/928)

* Add `qml.Projector` observable support via diagonalization to Lightning-GPU.
  [(#894)](https://github.com/PennyLaneAI/pennylane-lightning/pull/894)

* Add 1-target wire controlled gate support to `lightning.tensor`. Note that `cutensornet` only supports 1-target wire controlled gate as of `v24.08`. A controlled gate with more than 1 target wire should be converted to dense matrix.
  [(#880)](https://github.com/PennyLaneAI/pennylane-lightning/pull/880)

* Build and upload Lightning-Tensor wheels (x86_64, AARCH64) to PyPI.
  [(#862)](https://github.com/PennyLaneAI/pennylane-lightning/pull/862)
  [(#905)](https://github.com/PennyLaneAI/pennylane-lightning/pull/905)

* Add Matrix Product Operator (MPO) for all gates support to `lightning.tensor`. Note current C++ implementation only works for MPO sites data provided by users.
  [(#859)](https://github.com/PennyLaneAI/pennylane-lightning/pull/859)

* Add shots measurement support to `lightning.tensor`.
  [(#852)](https://github.com/PennyLaneAI/pennylane-lightning/pull/852)

* Lightning-GPU and Lightning-Kokkos migrated to the new device API.
  [(#853)](https://github.com/PennyLaneAI/pennylane-lightning/pull/853)
  [(#810)](https://github.com/PennyLaneAI/pennylane-lightning/pull/810)

### Breaking changes

* Update MacOS wheel build to 13.0 for X86_64 and ARM due to the deprecation of MacOS-12 CI runners.
  [(#969)](https://github.com/PennyLaneAI/pennylane-lightning/pull/969)

* Deprecate `initSV()` and add `resetStateVector()` from the C++ API Lightning-GPU. This is to remove the `reset_state` additional call in the Python layer.
  [(#933)](https://github.com/PennyLaneAI/pennylane-lightning/pull/933)

* Deprecate PI gates implementation in Lightning-Qubit. The PI gates were the first implementation of gate kernels in `lightning.qubit` using pre-computed indices, prior to the development of LM (less memory) and AVX kernels. This deprecation is in favour of reducing compilation time and ensuring that Lightning-Qubit only relies on LM kernels in the dynamic dispatcher across all platforms.
  [(#925)](https://github.com/PennyLaneAI/pennylane-lightning/pull/925)

* Remove PowerPC wheel build recipe for Lightning-Qubit.
  [(#902)](https://github.com/PennyLaneAI/pennylane-lightning/pull/902)

* Update MacOS wheel builds to require Monterey (12.0) or greater for x86_64 and ARM. This was required to update Pybind11 to the latest release (2.13.5) for enabling Numpy 2.0 support in Lightning.
  [(#901)](https://github.com/PennyLaneAI/pennylane-lightning/pull/901)

* Remove support for Python 3.9 for all Lightning simulators.
  [(#891)](https://github.com/PennyLaneAI/pennylane-lightning/pull/891)

### Improvements

* Update the `lightning.tensor` Python layer unit tests, as `lightning.tensor` cannot be cleaned up like other state-vector devices because the data is attached to the graph. It is recommended to use one device per circuit for `lightning.tensor`.
  [(#971)](https://github.com/PennyLaneAI/pennylane-lightning/pull/971)

* Add joint check for the N-controlled wires support in `lightning.qubit`.
  [(#949)](https://github.com/PennyLaneAI/pennylane-lightning/pull/949)

* Optimize `GlobalPhase` and `C(GlobalPhase)` gate implementation in `lightning.gpu`.
  [(#946)](https://github.com/PennyLaneAI/pennylane-lightning/pull/946)

* Add missing `liblightning_kokkos_catalyst.so` when building Lightning-Kokkos in editable installation.
  [(#945)](https://github.com/PennyLaneAI/pennylane-lightning/pull/945)

* Optimize the cartesian product to reduce the amount of memory necessary to set the `StatePrep` in Lightning-Tensor.
  [(#943)](https://github.com/PennyLaneAI/pennylane-lightning/pull/943)

* Update the `qml.probs` data-return in Lightning-GPU C++ API to align with other state-vector devices.
  [(#941)](https://github.com/PennyLaneAI/pennylane-lightning/pull/941)

* Add zero-state initialization to both `StateVectorCudaManaged` and `StateVectorCudaMPI` constructors to remove the `reset_state` in the Python layer ctor and refactor `setBasisState(state, wires)` in the C++ API.
  [(#933)](https://github.com/PennyLaneAI/pennylane-lightning/pull/933)

* Add `setStateVector(state, wire)` support to the Lightning-GPU C++ API.
  [(#930)](https://github.com/PennyLaneAI/pennylane-lightning/pull/930)

* The `generate_samples` methods of `lightning.qubit` and `lightning.kokkos` can now take in a seed number to make the generated samples deterministic. This can be useful when, among other things, fixing flaky tests in CI.
  [(#927)](https://github.com/PennyLaneAI/pennylane-lightning/pull/927)

* Remove dynamic decomposition rules for all Lightning devices.
  [(#926)](https://github.com/PennyLaneAI/pennylane-lightning/pull/926)

* Always decompose `qml.QFT` in all Lightning devices.
  [(#924)](https://github.com/PennyLaneAI/pennylane-lightning/pull/924)

* Uniform Python format to adhere PennyLane style.
  [(#924)](https://github.com/PennyLaneAI/pennylane-lightning/pull/924)

* Add the `ci:use-gpu-runner` GitHub label to `lightning.kokkos` GPU Testing CIs.
  [(#916)](https://github.com/PennyLaneAI/pennylane-lightning/pull/916)

* Update the test suite to remove deprecated code.
  [(#912)](https://github.com/PennyLaneAI/pennylane-lightning/pull/912)

* Merge `lightning.gpu` and `lightning.tensor` GPU tests in single Python and C++ CIs controlled by the `ci:use-gpu-runner` label.
  [(#911)](https://github.com/PennyLaneAI/pennylane-lightning/pull/911)

* Skip the compilation of Lightning simulators and development requirements to boost the build of public docs up to 5x.
  [(#904)](https://github.com/PennyLaneAI/pennylane-lightning/pull/904)

* Build Lightning wheels in `Release` mode to reduce the binary sizes.
  [(#903)](https://github.com/PennyLaneAI/pennylane-lightning/pull/903)

* Update Pybind11 to 2.13.5.
  [(#901)](https://github.com/PennyLaneAI/pennylane-lightning/pull/901)

* Migrate wheels artifacts to v4.
  [(#893)](https://github.com/PennyLaneAI/pennylane-lightning/pull/893)

* Update GitHub actions in response to a high-severity vulnerability.
  [(#887)](https://github.com/PennyLaneAI/pennylane-lightning/pull/887)

* Optimize and simplify controlled kernels in Lightning-Qubit.
  [(#882)](https://github.com/PennyLaneAI/pennylane-lightning/pull/882)

* Optimize gate cache recording for Lightning-Tensor C++ API.
  [(#879)](https://github.com/PennyLaneAI/pennylane-lightning/pull/879)

* Unify Lightning-Kokkos and Lightning-Qubit devices under a Lightning-Base abstracted class.
  [(#876)](https://github.com/PennyLaneAI/pennylane-lightning/pull/876)

* Smarter defaults for the `split_obs` argument in the serializer. The serializer splits linear combinations into chunks instead of all their terms.
  [(#873)](https://github.com/PennyLaneAI/pennylane-lightning/pull/873/)

* Prefer `tomlkit` over `toml` for building Lightning wheels, and choose `tomli` and `tomllib` over `toml` when installing the package.
  [(#857)](https://github.com/PennyLaneAI/pennylane-lightning/pull/857)

* Lightning-Kokkos gains native support for the `PauliRot` gate.
  [(#855)](https://github.com/PennyLaneAI/pennylane-lightning/pull/855)

### Documentation

* Update Lightning-Tensor installation docs and usage suggestions.
  [(#971)](https://github.com/PennyLaneAI/pennylane-lightning/pull/971)
  [(#972)](https://github.com/PennyLaneAI/pennylane-lightning/pull/971)

* Update `README.rst` installation instructions for `lightning.gpu` and `lightning.tensor`.
  [(#957)](https://github.com/PennyLaneAI/pennylane-lightning/pull/957)

* Update `lightning.tensor` documentation to include all the new features added since pull request #756. The new features are: 1. Finite-shot measurements; 2. Expval-base quantities; 3. Support for `qml.state()` and `qml.stateprep()`; 4. Support for all gates support via Matrix Product Operator (MPO).
  [(#909)](https://github.com/PennyLaneAI/pennylane-lightning/pull/909)

### Bug fixes

*  Fix Lightning Kokkos test_device for `kokkos_args` fail for MacOS due to `np.complex256`
  [(#974)](https://github.com/PennyLaneAI/pennylane-lightning/pull/974)

*  Fix PTM stable-latest related to `default.qubit.legacy` and the `latest` flag usage.
  [(#961)](https://github.com/PennyLaneAI/pennylane-lightning/pull/961)
  [(#966)](https://github.com/PennyLaneAI/pennylane-lightning/pull/966)

* Fix build failure for Lightning-Kokkos editable installation on MacOS due to `liblightning_kokkos_catalyst.so` copy and `liblightning_kokkos_catalyst.so` not copied to correct build path for editable installation.
  [(#947)](https://github.com/PennyLaneAI/pennylane-lightning/pull/947)
  [(#968)](https://github.com/PennyLaneAI/pennylane-lightning/pull/968)

* Add concept restriction to ensure `ConstMult` inline function only hit with arithmetic-values times complex values. Fixes build failures with the test suite when enabling OpenMP, and disabling BLAS and Python under clang.
  [(#936)](https://github.com/PennyLaneAI/pennylane-lightning/pull/936)

* Bug fix for `applyMatrix` in Lightning-Tensor. Matrix operator data is not stored in the `cuGateCache` object to support `TensorProd` obs with multiple `Hermitian` obs.
  [(#932)](https://github.com/PennyLaneAI/pennylane-lightning/pull/932)

* Bug fix for `_pauli_word` of `QuantumScriptSerializer`. `_pauli_word` can process `PauliWord` object: `I`.
  [(#919)](https://github.com/PennyLaneAI/pennylane-lightning/pull/919)

* Bug fix for analytic `qml.probs` in the Lightning-Tensor C++ API.
  [(#906)](https://github.com/PennyLaneAI/pennylane-lightning/pull/906)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Diego Guala, Joseph Lee, Luis Alfredo Nuñez Meneses, Vincent Michaud-Rioux, Lee J. O'Riordan, Mudit Pandey, Shuli Shu, Haochen Paul Wang

---

# Release 0.38.0

### New features since last release

* Add `qml.StatePrep()` and `qml.QubitStateVector()` support to `lightning.tensor`.
  [(#849)](https://github.com/PennyLaneAI/pennylane-lightning/pull/849)

* Add analytic `qml.probs()` measurement support to `lightning.tensor`.
  [(#830)](https://github.com/PennyLaneAI/pennylane-lightning/pull/830)

* Add `qml.state()` measurement support to `lightning.tensor`.
  [(#827)](https://github.com/PennyLaneAI/pennylane-lightning/pull/827)

* Add Lightning-GPU Linux (AArch64 + GraceHopper) wheels to PyPI.
  [(#815)](https://github.com/PennyLaneAI/pennylane-lightning/pull/815)

* Add `var` support to `lightning.tensor`. Note that `var` support is added via `obs**2` and this implementation scales as `O(num_obs**2)`.
  [(#804)](https://github.com/PennyLaneAI/pennylane-lightning/pull/804)

### Breaking changes

* Update python packaging to follow PEP 517/518/621/660 standards.
  [(#832)](https://github.com/PennyLaneAI/pennylane-lightning/pull/832)

* Add `getData()` in the `lightning.tensor` C++ backend. Users are responsible for ensuring sufficient host memory is allocated for the full state vector.
  [(#827)](https://github.com/PennyLaneAI/pennylane-lightning/pull/827)

* Remove `NDpermuter.hpp` which is no longer required.
  [(#795)](https://github.com/PennyLaneAI/pennylane-lightning/pull/795)

* Remove temporary steps from the CI, such as downgrading Scipy to <1.14 and installing Kokkos v4.2 for `lightning-version == 'stable'`.
  [(#792)](https://github.com/PennyLaneAI/pennylane-lightning/pull/792)

* Do not run GPU tests and Docker workflows on release.
  [(#788)](https://github.com/PennyLaneAI/pennylane-lightning/pull/788)

* Update python packaging to follow PEP 517/518/621/660 standards.
  [(#832)](https://github.com/PennyLaneAI/pennylane-lightning/pull/832)

### Improvements

* Updated calls of ``size_t`` to ``std::size_t`` everywhere.
  [(#816)](https://github.com/PennyLaneAI/pennylane-lightning/pull/816)

* Update Lightning tests to support the generalization of basis state preparation.
  [(#864)](https://github.com/PennyLaneAI/pennylane-lightning/pull/864)

* Add `SetState` and `SetBasisState` to `Lightning-KokkosSimulator`.
  [(#861)](https://github.com/PennyLaneAI/pennylane-lightning/pull/861)

* Remove use of the deprecated `Operator.expand` in favour of `Operator.decomposition`.
  [(#846)](https://github.com/PennyLaneAI/pennylane-lightning/pull/846)

* The `setBasisState` and `setStateVector` methods of `StateVectorLQubit` and `StateVectorKokkos` are overloaded to support PennyLane-like parameters.
  [(#843)](https://github.com/PennyLaneAI/pennylane-lightning/pull/843)

* Move `setBasisState`, `setStateVector` and `resetStateVector` from `StateVectorLQubitManaged` to `StateVectorLQubit`.
  [(#841)](https://github.com/PennyLaneAI/pennylane-lightning/pull/841)

* Update `generate_samples` in Lightning-Kokkos and Lightning-GPU to support `qml.measurements.Shots` type instances.
  [(#839)](https://github.com/PennyLaneAI/pennylane-lightning/pull/839)

* Add a Catalyst-specific wrapping class for Lightning Kokkos.
  [(#837)](https://github.com/PennyLaneAI/pennylane-lightning/pull/837)
  [(#770)](https://github.com/PennyLaneAI/pennylane-lightning/pull/770)

* Lightning-Qubit natively supports the `PauliRot` gate.
  [(#834)](https://github.com/PennyLaneAI/pennylane-lightning/pull/834)

* Multiple calls to the `append_mps_final_state()` API is allowed in `lightning.tensor`.
  [(#830)](https://github.com/PennyLaneAI/pennylane-lightning/pull/830)

* Add `initial_state_prep` option to Catalyst TOML file.
  [(#826)](https://github.com/PennyLaneAI/pennylane-lightning/pull/826)

* `ENABLE_LAPACK` is `OFF` by default for all Lightning backends.
  [(#825)](https://github.com/PennyLaneAI/pennylane-lightning/pull/825)

* Update `ctrl_decomp_zyz` tests with `len(control_wires) > 1`.
  [(#821)](https://github.com/PennyLaneAI/pennylane-lightning/pull/821)

* Update the Catalyst-specific wrapping class for Lightning Kokkos to track Catalyst's new support for MCM seeding.
  [(#819)](https://github.com/PennyLaneAI/pennylane-lightning/pull/819)

* Replace ``size_t`` by ``std::size_t`` everywhere.
  [(#816)](https://github.com/PennyLaneAI/pennylane-lightning/pull/816/)

* Shot batching is made more efficient by executing all the shots in one go on Lightning-Qubit.
  [(#814)](https://github.com/PennyLaneAI/pennylane-lightning/pull/814)

* Lightning-Qubit calls `generate_samples(wires)` on a minimal subset of wires when executing in finite-shot mode.
  [(#813)](https://github.com/PennyLaneAI/pennylane-lightning/pull/813)

* Update `LightingQubit.preprocess` to work with changes to preprocessing for mid-circuit measurements.
  [(#812)](https://github.com/PennyLaneAI/pennylane-lightning/pull/812)

* Avoid unnecessary memory reset in Lightning-Qubit's state vector class constructor.
  [(#811)](https://github.com/PennyLaneAI/pennylane-lightning/pull/811)

* Add `generate_samples(wires)` support in Lightning-Qubit, which samples faster for a subset of wires.
  [(#809)](https://github.com/PennyLaneAI/pennylane-lightning/pull/809)

* Optimize the OpenMP parallelization of Lightning-Qubit's `probs` for all number of targets.
  [(#807)](https://github.com/PennyLaneAI/pennylane-lightning/pull/807)

* Optimize `probs(wires)` of Lightning-Kokkos using various kernels. Which kernel is to be used depends on the device, number of qubits and number of target wires.
  [(#802)](https://github.com/PennyLaneAI/pennylane-lightning/pull/802)

* Add GPU device compute capability check for Lightning-Tensor.
  [(#803)](https://github.com/PennyLaneAI/pennylane-lightning/pull/803)

* Refactor CUDA utils Python bindings to a separate module.
  [(#801)](https://github.com/PennyLaneAI/pennylane-lightning/pull/801)

* Parallelize Lightning-Qubit `probs` with OpenMP when using the `-DLQ_ENABLE_KERNEL_OMP=1` CMake argument.
  [(#800)](https://github.com/PennyLaneAI/pennylane-lightning/pull/800)

* Implement `probs(wires)` using a bit-shift implementation akin to the gate kernels in Lightning-Qubit.
  [(#795)](https://github.com/PennyLaneAI/pennylane-lightning/pull/795)

* Enable setting the PennyLane version when invoking, for example, `make docker-build version=master pl_version=master`.
  [(#791)](https://github.com/PennyLaneAI/pennylane-lightning/pull/791)

### Documentation

* The installation instructions for all lightning plugins have been improved.
  [(#858)](https://github.com/PennyLaneAI/pennylane-lightning/pull/858)
  [(#851)](https://github.com/PennyLaneAI/pennylane-lightning/pull/851)

* Updated the README and added citation format for Lightning arXiv preprint.
  [(#818)](https://github.com/PennyLaneAI/pennylane-lightning/pull/818)

### Bug fixes

* Point to the right Lightning root folder independently from the invocation location, when configuring the project.
  [(#874)](https://github.com/PennyLaneAI/pennylane-lightning/pull/874)

* Update dependencies and `build` command options following changes in the build system.
  [(#863)](https://github.com/PennyLaneAI/pennylane-lightning/pull/863)

* Replace structured bindings by variables in `GateImplementationsLM.hpp`.
  [(#856)](https://github.com/PennyLaneAI/pennylane-lightning/pull/856)

* Remove wrong `-m` when calling `setup.py`.
  [(#854)](https://github.com/PennyLaneAI/pennylane-lightning/pull/854)

* Fix plugin-test-matrix CI/CD workflows.
  [(#850)](https://github.com/PennyLaneAI/pennylane-lightning/pull/850)

* Set the `immutable` parameter value as `false` for the `cutensornetStateApplyTensorOperator` to allow the following `cutensornetStateUpdateTensorOperator` call.
  [(#845)](https://github.com/PennyLaneAI/pennylane-lightning/pull/845)

* Fix cuQuantum SDK path pass-though in CMake.
  [(#831)](https://github.com/PennyLaneAI/pennylane-lightning/pull/831)

* Fix CUDA sync issues on AArch64 + GraceHopper.
  [(#823)](https://github.com/PennyLaneAI/pennylane-lightning/pull/823)

* Check for the number of wires for Hermitian observables in Lightning-Tensor. Only 1-wire Hermitian observables are supported as of `cuTensorNet-v24.03.0`.
  [(#806)](https://github.com/PennyLaneAI/pennylane-lightning/pull/806)

* Set `PL_BACKEND` for the entire `build-wheel-lightning-gpu` Docker-build stage to properly build the Lightning-GPU wheel.
  [(#791)](https://github.com/PennyLaneAI/pennylane-lightning/pull/791)

* Fix conditions for skipping build & push steps in the Docker build workflows.
  [(#790)](https://github.com/PennyLaneAI/pennylane-lightning/pull/790)

* Downgrade Scipy on Lightning stable version tests.
  [(#783)](https://github.com/PennyLaneAI/pennylane-lightning/pull/783)

* Fix checkout command in test workflows for rc branches.
  [(#777)](https://github.com/PennyLaneAI/pennylane-lightning/pull/777)

* Point to the right Lightning root folder independently from the invocation location, when configuring the project.
  [(#874)](https://github.com/PennyLaneAI/pennylane-lightning/pull/874)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Astral Cai, Ahmed Darwish, Amintor Dusko, Vincent Michaud-Rioux, Luis Alfredo Nuñez Meneses, Erick Ochoa Lopez, Lee J. O'Riordan, Mudit Pandey, Shuli Shu, Raul Torres, Paul Haochen Wang

---

# Release 0.37.0

### New features since last release

* Implement Python interface to the `lightning.tensor` device.
  [(#748)](https://github.com/PennyLaneAI/pennylane-lightning/pull/748)

* Add `inverse` support for gate operations in `lightning.tensor` in the C++ layer.
  [(#753)](https://github.com/PennyLaneAI/pennylane-lightning/pull/753)

* Add `observable` and `expval` support to the `cutensornet`-backed `lightning.tensor` C++ layer.
  [(#728)](https://github.com/PennyLaneAI/pennylane-lightning/pull/728)

* Add gate support to `cutensornet`-backed `lightning.tensor` C++ layer.
  [(#718)](https://github.com/PennyLaneAI/pennylane-lightning/pull/718)

* Add `cutensornet`-backed `MPS` C++ layer to `lightning.tensor`.
  [(#704)](https://github.com/PennyLaneAI/pennylane-lightning/pull/704)

* Add support for `C(BlockEncode)` to Lightning devices.
  [(#743)](https://github.com/PennyLaneAI/pennylane-lightning/pull/743)

### Breaking changes

* Removed the `QuimbMPS` class and the corresponding backend from `lightning.tensor`.
  [(#737)](https://github.com/PennyLaneAI/pennylane-lightning/pull/737)

* Changed the name of `default.tensor` to `lightning.tensor` with the `quimb` backend.
  [(#730)](https://github.com/PennyLaneAI/pennylane-lightning/pull/730)

* `dynamic_one_shot` uses shot-vectors in the auxiliary tape to tell the device how many times to repeat the tape. Lightning-Qubit is updated accordingly.
  [(#724)](https://github.com/PennyLaneAI/pennylane-lightning/pull/724)

* `dynamic_one_shot` deals with post-selection during the post-processing phase, so Lightning-Qubit does not return `None`-valued measurements for mismatching samples anymore.
  [(#720)](https://github.com/PennyLaneAI/pennylane-lightning/pull/720)

### Improvements

* Release candidate branches automatically use the new large GitHub runner pool.
  [(#769)](https://github.com/PennyLaneAI/pennylane-lightning/pull/769)

* Lightning-Kokkos dev wheels for MacOS (x86_64, ARM64) and Linux (AArch64) are uploaded to TestPyPI upon merging a pull request.
  [(#765)](https://github.com/PennyLaneAI/pennylane-lightning/pull/765)

* Lightning-Kokkos Linux (x86_64) dev wheels are pushed to [Test PyPI](https://test.pypi.org/project/PennyLane-Lightning-Kokkos/) upon merging a pull request.
  [(#763)](https://github.com/PennyLaneAI/pennylane-lightning/pull/763)

* Change the type of tensor network objects passed to `ObservablesTNCuda` and `MeasurementsTNCuda` classes from `StateTensorT` to `TensorNetT`.
  [(#759)](https://github.com/PennyLaneAI/pennylane-lightning/pull/759)

* Silence `NDPermuter` linting warnings.
  [(#750)](https://github.com/PennyLaneAI/pennylane-lightning/pull/750)

* Rationalize MCM tests, removing most end-to-end tests from the native MCM test file, but keeping one that validates multiple mid-circuit measurements with any allowed return.
  [(#754)](https://github.com/PennyLaneAI/pennylane/pull/754)

* Rename `lightning.tensor` C++ libraries.
  [(#755)](https://github.com/PennyLaneAI/pennylane-lightning/pull/755)

* Set `state_tensor` as `const` for the `MeasurementTNCuda` class.
  [(#753)](https://github.com/PennyLaneAI/pennylane-lightning/pull/753)

* Updated Kokkos version and support to 4.3.01.
  [(#725)](https://github.com/PennyLaneAI/pennylane-lightning/pull/725)

* Lightning-Kokkos' functors are rewritten as functions wrapping around generic gate and generator functors templated over a coefficient interaction function. This reduces boilerplate while clarifying how the various kernels differ from one another.
  [(#640)](https://github.com/PennyLaneAI/pennylane-lightning/pull/640)

* Update C++ and Python GitHub actions names to include the matrix info.
  [(#717)](https://github.com/PennyLaneAI/pennylane-lightning/pull/717)

* Remove `CPhase` in favour of `CPhaseShift` in Lightning devices.
  [(#717)](https://github.com/PennyLaneAI/pennylane-lightning/pull/717)

* The various OpenMP configurations of Lightning-Qubit are tested in parallel on different Github Actions runners.
  [(#712)](https://github.com/PennyLaneAI/pennylane-lightning/pull/712)

* Update Linux wheels to use `manylinux_2_28` images.
  [(#667)](https://github.com/PennyLaneAI/pennylane-lightning/pull/667)

* Add support for `qml.expval` and `qml.var` in the `lightning.tensor` device for the `quimb` interface and the MPS method.
  [(#686)](https://github.com/PennyLaneAI/pennylane-lightning/pull/686)

* Changed the name of `lightning.tensor` to `default.tensor` with the `quimb` backend.
  [(#719)](https://github.com/PennyLaneAI/pennylane-lightning/pull/719)

* `lightning.qubit` and `lightning.kokkos` adhere to user-specified mid-circuit measurement configuration options.
  [(#736)](https://github.com/PennyLaneAI/pennylane-lightning/pull/736)

* Patch the C++ `Measurements.probs(wires)` method in Lightning-Qubit and Lightning-Kokkos to `Measurements.probs()` when called with all wires.
  This will trigger a more optimized implementation for calculating the probabilities of the entire system.
  [(#744)](https://github.com/PennyLaneAI/pennylane-lightning/pull/744)

* Remove the daily schedule from the "Compat Check w/PL - release/release" GitHub action.
  [(#746)](https://github.com/PennyLaneAI/pennylane-lightning/pull/746)

* Remove the required `scipy` config file for Lightning-Qubit. The support is now maintained by passing `SCIPY_LIBS_PATH` to the compiler.
  [(#775)](https://github.com/PennyLaneAI/pennylane-lightning/pull/775)

### Documentation

* Add installation instructions and documentation for `lightning.tensor`.
  [(#756)](https://github.com/PennyLaneAI/pennylane-lightning/pull/756)

### Bug fixes

* Don't route `probs(wires=[])` to `probs(all_wires)` in Lightning-Kokkos.
  [(#762)](https://github.com/PennyLaneAI/pennylane-lightning/pull/762)

* `ControlledQubitUnitary` is present in the Python device but not the TOML files. It is added to the decomposition gates since it can be implemented in its alternate form of `C(QubitUnitary)`.
  [(#767)](https://github.com/PennyLaneAI/pennylane-lightning/pull/767)

* Update the Lightning TOML files to indicate that non-commuting observables are supported.
  [(#764)](https://github.com/PennyLaneAI/pennylane-lightning/pull/764)

* Fix regex matching issue with auto on-boarding of release candidate branch to using the large runner queue.
  [(#774)](https://github.com/PennyLaneAI/pennylane-lightning/pull/774)

* Fix random CI failures for `lightning.tensor` Python unit tests and ignore `lightning_tensor` paths.
  [(#761)](https://github.com/PennyLaneAI/pennylane-lightning/pull/761)

* `lightning.qubit` and `lightning.kokkos` use `qml.ops.Conditional.base` instead of `qml.ops.Conditional.then_op`.
  [(#752)](https://github.com/PennyLaneAI/pennylane-lightning/pull/752)

* The preprocessing step in `lightning.qubit` now uses interface information to properly support the hardware-like postselection for mid-circuit measurements.
  [(#760)](https://github.com/PennyLaneAI/pennylane-lightning/pull/760)

* Fix AVX streaming operation support with newer GCC.
  [(#729)](https://github.com/PennyLaneAI/pennylane-lightning/pull/729)

* Revert changes calling the templated `IMAG`, `ONE`, `ZERO` functions in Kokkos kernels since they are incompatible with device execution.
  [(#733)](https://github.com/PennyLaneAI/pennylane-lightning/pull/733)

* The `tests_lkcpu_python.yml` workflow properly checks out the release or stable version of Lightning-Qubit during the test job.
  [(#723)](https://github.com/PennyLaneAI/pennylane-lightning/pull/723)

* Fix PennyLane Lightning-Kokkos and Lightning-Qubit tests for stable/stable configuration.
  [(#734)](https://github.com/PennyLaneAI/pennylane-lightning/pull/734)

* Remove the Autoray dependency from requirement files.
  [(#736)](https://github.com/PennyLaneAI/pennylane-lightning/pull/736)

* Fix the `cuda-runtime-12-0` dependency issue on RHEL8.
  [(#739)](https://github.com/PennyLaneAI/pennylane-lightning/pull/739)

* Fix the memory segmentation fault when initializing zero-wire Lightning-Kokkos.
  [(#757)](https://github.com/PennyLaneAI/pennylane-lightning/pull/757)

* Remove `pennylane.ops.op_math.controlled_decompositions.ctrl_decomp_zyz` tests with `len(control_wires) > 1`.
  [(#757)](https://github.com/PennyLaneAI/pennylane-lightning/pull/757)

* Add support for Scipy v1.14.
  [(#776)](https://github.com/PennyLaneAI/pennylane-lightning/pull/776)

* Add pickle support for the `DevPool` object in `lightning.gpu`.
  [(#772)](https://github.com/PennyLaneAI/pennylane-lightning/pull/772)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Lillian Frederiksen, Pietropaolo Frisoni, David Ittah, Vincent Michaud-Rioux, Lee James O'Riordan, Mudit Pandey, Shuli Shu, Jay Soni

---

# Release 0.36.0

### New features since last release

* Add `cutensornet`-backed `MPS` C++ layer to `lightning.tensor`.
  [(#704)](https://github.com/PennyLaneAI/pennylane-lightning/pull/704)

* Add Python class for the `lightning.tensor` device which uses the new device API and the interface for `quimb` based on the MPS method.
  [(#671)](https://github.com/PennyLaneAI/pennylane-lightning/pull/671)

* Add compile-time support for AVX2/512 streaming operations in `lightning.qubit`.
  [(#664)](https://github.com/PennyLaneAI/pennylane-lightning/pull/664)

* `lightning.kokkos` supports mid-circuit measurements.
  [(#672)](https://github.com/PennyLaneAI/pennylane-lightning/pull/672)

* Add dynamic linking to LAPACK/OpenBlas shared objects in `scipy.libs` for both C++ and Python layer.
  [(#653)](https://github.com/PennyLaneAI/pennylane-lightning/pull/653)

* `lightning.qubit` supports mid-circuit measurements.
  [(#650)](https://github.com/PennyLaneAI/pennylane-lightning/pull/650)

* Add finite shots support in `lightning.qubit2`.
  [(#630)](https://github.com/PennyLaneAI/pennylane-lightning/pull/630)

* Add `collapse` and `normalize` methods to the `StateVectorLQubit` classes, enabling "branching" of the wavefunction. Add methods to create and seed an RNG in the `Measurements` modules.
  [(#645)](https://github.com/PennyLaneAI/pennylane-lightning/pull/645)

* Add two new Python classes (LightningStateVector and LightningMeasurements) to support `lightning.qubit2`.
  [(#613)](https://github.com/PennyLaneAI/pennylane-lightning/pull/613)

* Add analytic-mode `qml.probs` and `qml.var` support in `lightning.qubit2`.
  [(#627)](https://github.com/PennyLaneAI/pennylane-lightning/pull/627)

* Add `LightningAdjointJacobian` to support `lightning.qubit2`.
  [(#631)](https://github.com/PennyLaneAI/pennylane-lightning/pull/631)

* Add `lightning.qubit2` device which uses the new device API.
  [(#607)](https://github.com/PennyLaneAI/pennylane-lightning/pull/607)
  [(#628)](https://github.com/PennyLaneAI/pennylane-lightning/pull/628)

* Add Vector-Jacobian Product calculation support to `lightning.qubit`.
  [(#644)](https://github.com/PennyLaneAI/pennylane-lightning/pull/644)

* Add support for using new operator arithmetic as the default.
  [(#649)](https://github.com/PennyLaneAI/pennylane-lightning/pull/649)

### Breaking changes

* Split Lightning-Qubit and Lightning-Kokkos CPU Python tests with `pytest-split`. Remove `SERIAL` from Kokkos' `exec_model` matrix. Remove `all` from Lightning-Kokkos' `pl_backend` matrix. Move `clang-tidy` checks to `tidy.yml`. Avoid editable `pip` installations.
  [(#696)](https://github.com/PennyLaneAI/pennylane-lightning/pull/696)
* Update `lightning.gpu` and `lightning.kokkos` to raise an error instead of falling back to `default.qubit`.
  [(#689)](https://github.com/PennyLaneAI/pennylane-lightning/pull/689)

* Add `paths` directives to test workflows to avoid running tests that cannot be impacted by changes.
  [(#699)](https://github.com/PennyLaneAI/pennylane-lightning/pull/699)
  [(#695)](https://github.com/PennyLaneAI/pennylane-lightning/pull/695)

* Move common components of `/src/simulator/lightning_gpu/utils/` to `/src/utils/cuda_utils/`.
  [(#676)](https://github.com/PennyLaneAI/pennylane-lightning/pull/676)

* Deprecate static LAPACK linking support.
  [(#653)](https://github.com/PennyLaneAI/pennylane-lightning/pull/653)

* Migrate `lightning.qubit` to the new device API.
  [(#646)](https://github.com/PennyLaneAI/pennylane-lightning/pull/646)

* Introduce `ci:build_wheels` label, which controls wheel building on `pull_request` and other triggers.
  [(#648)](https://github.com/PennyLaneAI/pennylane-lightning/pull/648)

* Remove building wheels for Lightning Kokkos on Windows.
  [(#693)](https://github.com/PennyLaneAI/pennylane-lightning/pull/693)

### Improvements

* Add tests for Windows Wheels, fix ill-defined caching, and set the proper backend for `lightning.kokkos` wheels.
  [(#693)](https://github.com/PennyLaneAI/pennylane-lightning/pull/693)

* Replace string comparisons by `isinstance` checks where possible.
  [(#691)](https://github.com/PennyLaneAI/pennylane-lightning/pull/691)

* Refactor `cuda_utils` to remove its dependency on `custatevec.h`.
  [(#681)](https://github.com/PennyLaneAI/pennylane-lightning/pull/681)

* Add `test_templates.py` module where Grover and QSVT are tested.
  [(#684)](https://github.com/PennyLaneAI/pennylane-lightning/pull/684)

* Create `cuda_utils` for common usage of CUDA related backends.
  [(#676)](https://github.com/PennyLaneAI/pennylane-lightning/pull/676)

* Refactor `lightning_gpu_utils` unit tests to remove the dependency on statevector class.
  [(#675)](https://github.com/PennyLaneAI/pennylane-lightning/pull/675)

* Upgrade GitHub actions versions from v3 to v4.
  [(#669)](https://github.com/PennyLaneAI/pennylane-lightning/pull/669)

* Initialize the private attributes `gates_indices_` and `generators_indices_` of `StateVectorKokkos` using the definitions of the `Pennylane::Gates::Constant` namespace.
  [(#641)](https://github.com/PennyLaneAI/pennylane-lightning/pull/641)

* Add `isort` to `requirements-dev.txt` and run before `black` upon `make format` to sort Python imports.
  [(#623)](https://github.com/PennyLaneAI/pennylane-lightning/pull/623)

* Improve support for new operator arithmetic with `QuantumScriptSerializer.serialize_observables`.
  [(#670)](https://github.com/PennyLaneAI/pennylane-lightning/pull/670)

* Add `workflow_dispatch` to wheels recipes; allowing developers to build wheels manually on a branch instead of temporarily changing the headers.
  [(#679)](https://github.com/PennyLaneAI/pennylane-lightning/pull/679)

* Add the `ENABLE_LAPACK` compilation flag to toggle dynamic linking to LAPACK library.
  [(#678)](https://github.com/PennyLaneAI/pennylane-lightning/pull/678)

### Documentation

### Bug fixes

* Fix wire order permutations when using `qml.probs` with out-of-order wires in Lightning-Qubit.
  [(#707)](https://github.com/PennyLaneAI/pennylane-lightning/pull/707)

* Lightning-Qubit once again respects the wire order specified on device instantiation.
  [(#705)](https://github.com/PennyLaneAI/pennylane-lightning/pull/705)

* `dynamic_one_shot` was refactored to use `SampleMP` measurements as a way to return the mid-circuit measurement samples. `LightningQubit's `simulate` is modified accordingly.
  [(#694)](https://github.com/PennyLaneAI/pennylane-lightning/pull/694)

* Lightning-Qubit correctly decomposes state prep operations when used in the middle of a circuit.
  [(#687)](https://github.com/PennyLaneAI/pennylane-lightning/pull/687)

* Lightning-Qubit correctly decomposes `qml.QFT` and `qml.GroverOperator` if `len(wires)` is greater than 9 and 12 respectively.
  [(#687)](https://github.com/PennyLaneAI/pennylane-lightning/pull/687)

* Specify `isort` `--py` (Python version) and `-l` (max line length) to stabilize `isort` across Python versions and environments.
  [(#647)](https://github.com/PennyLaneAI/pennylane-lightning/pull/647)

* Fix random `coverage xml` CI issues.
  [(#635)](https://github.com/PennyLaneAI/pennylane-lightning/pull/635)

* `lightning.qubit` correctly decomposed state preparation operations with adjoint differentiation.
  [(#661)](https://github.com/PennyLaneAI/pennylane-lightning/pull/661)

* Fix the failed observable serialization unit tests.
  [(#683)](https://github.com/PennyLaneAI/pennylane-lightning/pull/683)

* Update the Lightning-Qubit new device API to work with Catalyst.
  [(#665)](https://github.com/PennyLaneAI/pennylane-lightning/pull/665)

* Update the version of `codecov-action` to v4 and fix the CodeCov issue with the PL-Lightning check-compatibility actions.
  [(#682)](https://github.com/PennyLaneAI/pennylane-lightning/pull/682)

* Refactor of dev prerelease auto-update-version workflow.
  [(#685)](https://github.com/PennyLaneAI/pennylane-lightning/pull/685)

* Remove gates unsupported by catalyst from TOML file.
  [(#698)](https://github.com/PennyLaneAI/pennylane-lightning/pull/698)

* Increase tolerance for a flaky test.
  [(#703)](https://github.com/PennyLaneAI/pennylane-lightning/pull/703)

* Remove `ControlledQubitUnitary` in favour of `C(QubitUnitary)` from the list of supported operations and the device TOML file. The `stopping_condition` method guarantees the consistency of decompositions.
  [(#758)](https://github.com/PennyLaneAI/pennylane-lightning/pull/758)

* Raise a clear error message with initialization of `lightning.kokkos` with zero-qubit on Windows.
  [(#758)](https://github.com/PennyLaneAI/pennylane-lightning/pull/758)


### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Pietropaolo Frisoni, Thomas Germain, Christina Lee, Erick Ochoa Lopez, Vincent Michaud-Rioux, Rashid N H M, Lee James O'Riordan, Mudit Pandey, Shuli Shu

---

# Release 0.35.1

### Improvements

* Use the `adjoint` gate parameter to apply `qml.Adjoint` operations instead of matrix methods in `lightning.qubit`.
  [(#632)](https://github.com/PennyLaneAI/pennylane-lightning/pull/632)

### Bug fixes

* Fix `qml.Adjoint` support in `lightning.gpu` and `lightning.kokkos`.
  [(#632)](https://github.com/PennyLaneAI/pennylane-lightning/pull/632)

* Fix finite shots support in `lightning.qubit`, `lightning.gpu` and `lightning.kokkos`. The bug would impact calculations with measurements on observables with non-trivial diagonalizing gates and calculations with shot vectors.
  [(#632)](https://github.com/PennyLaneAI/pennylane-lightning/pull/632)

### Contributors

This release contains contributions from (in alphabetical order):

Vincent Michaud-Rioux

---

# Release 0.35.0

### New features since last release

* All backends now support `GlobalPhase` and `C(GlobalPhase)` in forward pass.
  [(#579)](https://github.com/PennyLaneAI/pennylane-lightning/pull/579)

* Add Hermitian observable support for shot-noise measurement and Lapack support.
  [(#569)](https://github.com/PennyLaneAI/pennylane-lightning/pull/569)

### Breaking changes

* Migrate `lightning.gpu` to CUDA 12.
  [(#606)](https://github.com/PennyLaneAI/pennylane-lightning/pull/606)

### Improvements

* Expand error values and strings returned from CUDA libraries.
  [(#617)](https://github.com/PennyLaneAI/pennylane-lightning/pull/617)

* `C(MultiRZ)` and `C(Rot)` gates are natively supported (with `LM` kernels).
  [(#614)](https://github.com/PennyLaneAI/pennylane-lightning/pull/614)

* Add adjoint support for `GlobalPhase` in Lightning-GPU and Lightning-Kokkos.
  [(#615)](https://github.com/PennyLaneAI/pennylane-lightning/pull/615)

* Lower the overheads of Windows CI tests.
  [(#610)](https://github.com/PennyLaneAI/pennylane-lightning/pull/610)

* Decouple LightningQubit memory ownership from numpy and migrate it to Lightning-Qubit managed state-vector class.
  [(#601)](https://github.com/PennyLaneAI/pennylane-lightning/pull/601)

* Expand support for Projector observables on Lightning-Kokkos.
  [(#601)](https://github.com/PennyLaneAI/pennylane-lightning/pull/601)

* Split Docker build cron job into two jobs: master and latest. This is mainly for reporting in the `plugin-test-matrix` repo.
  [(#600)](https://github.com/PennyLaneAI/pennylane-lightning/pull/600)

* The `BlockEncode` operation from PennyLane is now supported on all Lightning devices.
  [(#599)](https://github.com/PennyLaneAI/pennylane-lightning/pull/599)

* OpenMP acceleration can now be enabled at compile time for all `lightning.qubit` gate kernels using the `-DLQ_ENABLE_KERNEL_OMP=1` CMake argument.
  [(#510)](https://github.com/PennyLaneAI/pennylane-lightning/pull/510)

* Enable building Docker images for any branch or tag. Set the Docker build cron job to build images for the latest release and `master`.
  [(#598)](https://github.com/PennyLaneAI/pennylane-lightning/pull/598)

* Enable choosing the PennyLane-Lightning version and disabling push to Docker Hub in the Docker build workflow. Add a cron job calling the Docker build workflow.
  [(#597)](https://github.com/PennyLaneAI/pennylane-lightning/pull/597)

* Pull Kokkos v4.2.00 from the official Kokkos repository to test Lightning-Kokkos with the CUDA backend.
  [(#596)](https://github.com/PennyLaneAI/pennylane-lightning/pull/596)

* Remove deprecated MeasurementProcess.name.
  [(#605)](https://github.com/PennyLaneAI/pennylane-lightning/pull/605)

### Documentation

* Update requirements to build the documentation.
  [(#594)](https://github.com/PennyLaneAI/pennylane-lightning/pull/594)

### Bug fixes

* Downgrade auditwheel due to changes with library exclusion list.
  [(#620)](https://github.com/PennyLaneAI/pennylane-lightning/pull/620)

* List `GlobalPhase` gate in each device's TOML file.
  [(#615)](https://github.com/PennyLaneAI/pennylane-lightning/pull/615)

* Lightning-GPU's gate cache failed to distinguish between certain gates.
  For example, `MultiControlledX([0, 1, 2], "111")` and `MultiControlledX([0, 2], "00")` were applied as the same operation.
  This could happen with (at least) the following gates: `QubitUnitary`,`ControlledQubitUnitary`,`MultiControlledX`,`DiagonalQubitUnitary`,`PSWAP`,`OrbitalRotation`.
  [(#579)](https://github.com/PennyLaneAI/pennylane-lightning/pull/579)

* Ensure the stopping condition decompositions are respected for larger templated QFT and Grover operators.
  [(#609)](https://github.com/PennyLaneAI/pennylane-lightning/pull/609)

* Move concurrency group specifications from reusable Docker build workflow to the root workflows.
  [(#604)](https://github.com/PennyLaneAI/pennylane-lightning/pull/604)

* Fix `lightning-kokkos-cuda` Docker build and add CI workflow to build images and push to Docker Hub.
  [(#593)](https://github.com/PennyLaneAI/pennylane-lightning/pull/593)

* Update jax.config imports.
  [(#619)](https://github.com/PennyLaneAI/pennylane-lightning/pull/619)

* Fix apply state vector when using a Lightning handle.
  [(#622)](https://github.com/PennyLaneAI/pennylane-lightning/pull/622)

* Pinning Pytest to a version compatible with Flaky.
  [(#624)](https://github.com/PennyLaneAI/pennylane-lightning/pull/624)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, David Ittah, Vincent Michaud-Rioux, Lee J. O'Riordan, Shuli Shu, Matthew Silverman

---

# Release 0.34.0

### New features since last release

* Support added for Python 3.12 wheel builds.
  [(#541)](https://github.com/PennyLaneAI/pennylane-lightning/pull/541)

* Lightning-Qubit support arbitrary controlled gates (any wires and any control values). The kernels are implemented in the `LM` module.
  [(#576)](https://github.com/PennyLaneAI/pennylane-lightning/pull/576)

* Shot-noise related methods now accommodate observable objects with arbitrary eigenvalues. Add a Kronecker product method for two diagonal matrices.
  [(#570)](https://github.com/PennyLaneAI/pennylane-lightning/pull/570)

* Add shot-noise support for probs in the C++ layer. Probabilities are calculated from generated samples. All Lightning backends support this feature. Please note that target wires should be sorted in ascending manner.
  [(#568)](https://github.com/PennyLaneAI/pennylane-lightning/pull/568)

* Add `LM` kernels to apply arbitrary controlled operations efficiently.
  [(#516)](https://github.com/PennyLaneAI/pennylane-lightning/pull/516)

* Add shots support for variance value, probs, sample, counts calculation for given observables (`NamedObs`, `TensorProd` and `Hamiltonian`) based on Pauli words, `Identity` and `Hadamard` in the C++ layer. All Lightning backends support this support feature.
  [(#561)](https://github.com/PennyLaneAI/pennylane-lightning/pull/561)

* Add shots support for expectation value calculation for given observables (`NamedObs`, `TensorProd` and `Hamiltonian`) based on Pauli words, `Identity` and `Hadamard` in the C++ layer by adding `measure_with_samples` to the measurement interface. All Lightning backends support this support feature.
  [(#556)](https://github.com/PennyLaneAI/pennylane-lightning/pull/556)

* `qml.QubitUnitary` operators can be included in a circuit differentiated with the adjoint method. Lightning handles circuits with arbitrary non-differentiable `qml.QubitUnitary` operators. 1,2-qubit `qml.QubitUnitary` operators with differentiable parameters can be differentiated using decomposition.
  [(#540)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/540)

### Breaking changes

* Set the default version of Kokkos to 4.2.00 throughout the project (CMake, CI, etc.)
  [(#578)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/578)

* Overload `applyOperation` with a fifth `matrix` argument to all state vector classes to support arbitrary operations in `AdjointJacobianBase`.
  [(#540)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/540)

### Improvements

* Ensure aligned memory used for numpy arrays with state-vector without reallocations.
  [(#572)](https://github.com/PennyLaneAI/pennylane-lightning/pull/572)

* Unify error messages of shot measurement related unsupported observables to better Catalyst.
  [(#577)](https://github.com/PennyLaneAI/pennylane-lightning/pull/577)

* Add configuration files to improve compatibility with Catalyst.
  [(#566)](https://github.com/PennyLaneAI/pennylane-lightning/pull/566)

* Refactor shot-noise related methods of MeasurementsBase class in the C++ layer and eigenvalues are not limited to `1` and `-1`. Add `getObs()` method to Observables class. Refactor `applyInPlaceShots` to allow users to get eigenvalues of Observables object. Deprecated `_preprocess_state` method in `MeasurementsBase` class for safer use of the `LightningQubitRaw` backend.
[(#570)](https://github.com/PennyLaneAI/pennylane-lightning/pull/570)

* Modify `setup.py` to use backend-specific build directory (`f"build_{backend}"`) to accelerate rebuilding backends in alternance.
  [(#540)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/540)

* Update Dockerfile and rewrite the `build-wheel-lightning-gpu` stage to build Lightning-GPU from the `pennylane-lightning` monorepo.
  [(#539)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/539)

* Add the MPI test CI workflows of Lightning-GPU in compatibility cron jobs.
  [(#536)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/536)

* Add MPI synchronization in places to safely handle communicated data.
  [(#538)](https://github.com/PennyLaneAI/pennylane-lightning/pull/538)

* Add release option in compatibility cron jobs to test the release candidates of PennyLane and the Lightning plugins against one another.
  [(#531)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/531)

* Add GPU workflows in compatibility cron jobs to test Lightning-GPU and Lightning-Kokkos with the Kokkos CUDA backend.
  [(#528)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/528)

### Documentation

* Fixed a small typo in the documentation page for the PennyLane-Lightning GPU device.
  [(#563)](https://github.com/PennyLaneAI/pennylane-lightning/pull/563)

* Add OpenGraph social preview for Lightning docs.
  [(#574)](https://github.com/PennyLaneAI/pennylane-lightning/pull/574)

### Bug fixes

* Fix CodeCov file contention issue when uploading data from many workloads.
  [(#584)](https://github.com/PennyLaneAI/pennylane-lightning/pull/584)

* Ensure the `lightning.gpu` intermediate wheel builds are uploaded to TestPyPI.
  [(#575)](https://github.com/PennyLaneAI/pennylane-lightning/pull/575)

* Allow support for newer clang-tidy versions on non-x86_64 platforms.
  [(#567)](https://github.com/PennyLaneAI/pennylane-lightning/pull/567)

* Do not run C++ tests when testing for compatibility with PennyLane, hence fixing plugin-matrix failures. Fix Lightning-GPU workflow trigger.
  [(#571)](https://github.com/PennyLaneAI/pennylane-lightning/pull/571)

* Revert single-node multi-GPU batching behaviour to match https://github.com/PennyLaneAI/pennylane-lightning-gpu/pull/27.
  [(#564)](https://github.com/PennyLaneAI/pennylane-lightning/pull/564)

* Move deprecated `stateprep` `QuantumScript` argument into the operation list in `mpitests/test_adjoint_jacobian.py`.
  [(#540)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/540)

* Fix MPI Python unit tests for the adjoint method.
  [(#538)](https://github.com/PennyLaneAI/pennylane-lightning/pull/538)

* Fix the issue with assigning kernels to ops before registering kernels on macOS
  [(#582)](https://github.com/PennyLaneAI/pennylane-lightning/pull/582)

* Update `MANIFEST.in` to include device config files and `CHANGELOG.md`
  [(#585)](https://github.com/PennyLaneAI/pennylane-lightning/pull/585)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Isaac De Vlugt, Amintor Dusko, Vincent Michaud-Rioux, Erick Ochoa Lopez, Lee James O'Riordan, Shuli Shu

---

# Release 0.33.1

* pip-installed CUDA runtime libraries can now be accessed from a virtualenv.
  [(#543)](https://github.com/PennyLaneAI/pennylane-lightning/pull/543)

### Bug fixes

* The pybind11 compiled module RPATH linkage has been restored to pre-0.33 behaviour.
  [(#543)](https://github.com/PennyLaneAI/pennylane-lightning/pull/543)

### Contributors

This release contains contributions from (in alphabetical order):

Lee J. O'Riordan

---

# Release 0.33.0

### New features since last release

* Add documentation updates for the `lightning.gpu` backend.
  [(#525)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/525)

* Add `SparseHamiltonian` support for Lightning-Qubit and Lightning-GPU.
  [(#526)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/526)

* Add `SparseHamiltonian` support for Lightning-Kokkos.
  [(#527)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/527)

* Integrate python/pybind layer of distributed Lightning-GPU into the Lightning monorepo with Python unit tests.
  [(#518)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/518)

* Integrate the distributed C++ backend of Lightning-GPU into the Lightning monorepo.
  [(#514)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/514)

* Integrate Lightning-GPU into the Lightning monorepo. The new backend is named `lightning.gpu` and includes all single-GPU features.
  [(#499)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/499)

* Build Linux wheels for Lightning-GPU (CUDA-11).
  [(#517)](https://github.com/PennyLaneAI/pennylane-lightning/pull/517)

* Add `Dockerfile` in `docker` and `make docker` workflow in `Makefile`. The Docker images and documentation are available on [DockerHub](https://hub.docker.com/repository/docker/pennylaneai/pennylane).
  [(#496)](https://github.com/PennyLaneAI/pennylane-lightning/pull/496)

* Add mid-circuit state preparation operation tests.
  [(#495)](https://github.com/PennyLaneAI/pennylane-lightning/pull/495)

### Breaking changes

* Add `tests_gpu.yml` workflow to test the Lightning-Kokkos backend with CUDA-12.
  [(#494)](https://github.com/PennyLaneAI/pennylane-lightning/pull/494)

* Implement `LM::GeneratorDoubleExcitation`, `LM::GeneratorDoubleExcitationMinus`, `LM::GeneratorDoubleExcitationPlus` kernels. Lightning-Qubit default kernels are now strictly from the `LM` implementation, which requires less memory and is faster for large state vectors.
  [(#512)](https://github.com/PennyLaneAI/pennylane-lightning/pull/512)

* Add workflows validating compatibility between PennyLane and Lightning's most recent stable releases and development (latest) versions.
  [(#507)](https://github.com/PennyLaneAI/pennylane-lightning/pull/507)
  [(#498)](https://github.com/PennyLaneAI/pennylane-lightning/pull/498)

* Introduce `timeout-minutes` in various workflows, mainly to avoid Windows builds hanging for several hours.
  [(#503)](https://github.com/PennyLaneAI/pennylane-lightning/pull/503)

* Cast integral-valued arrays to the device's complex type on entry in `_preprocess_state_vector` to ensure the state is correctly represented with floating-point numbers.
  [(#501)](https://github.com/PennyLaneAI/pennylane-lightning/pull/501)

* Update `DefaultQubit` to `DefaultQubitLegacy` on Lightning fallback.
  [(#500)](https://github.com/PennyLaneAI/pennylane-lightning/pull/500)

* Enums defined in `GateOperation.hpp` start at `1` (previously `0`). `::BEGIN` is introduced in a few places where it was assumed `0` accordingly.
  [(#485)](https://github.com/PennyLaneAI/pennylane-lightning/pull/485)

* Enable pre-commit hooks to format all Python files and linting of all Python source files.
  [(#485)](https://github.com/PennyLaneAI/pennylane-lightning/pull/485)

### Improvements

* Improve Python testing for Lightning-GPU (+MPI) by adding jobs in Actions files and adding Python tests to increase code coverage.
  [(#522)](https://github.com/PennyLaneAI/pennylane-lightning/pull/522)

* Add support for `pip install pennylane-lightning[kokkos]` for the OpenMP backend.
  [(#515)](https://github.com/PennyLaneAI/pennylane-lightning/pull/515)

* Update `setup.py` to allow for multi-package co-existence. The `PennyLane_Lightning` package now is the responsible for the core functionality, and will be depended upon by all other extensions.
  [(#504)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/504)

* Redesign Lightning-Kokkos `StateVectorKokkos` class to use Kokkos `RangePolicy` together with special functors in `applyMultiQubitOp` to apply 1- to 4-wire generic unitary gates. For more than 4 wires, the general implementation using Kokkos `TeamPolicy` is employed to yield the best all-around performance.
  [(#490)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/490)

* Redesign Lightning-Kokkos `Measurements` class to use Kokkos `RangePolicy` together with special functors to obtain the expectation value of 1- to 4-wire generic unitary gates. For more than 4 wires, the general implementation using Kokkos `TeamPolicy` is employed to yield the best all-around performance.
  [(#489)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/489)

* Add tests to increase Lightning-Kokkos coverage.
  [(#485)](https://github.com/PennyLaneAI/pennylane-lightning/pull/485)

* Add memory locality tag reporting and adjoint diff dispatch for `lightning.qubit` statevector classes.
  [(#492)](https://github.com/PennyLaneAI/pennylane-lightning/pull/492)

* Add support for dependent external packages to C++ core.
  [(#482)](https://github.com/PennyLaneAI/pennylane-lightning/pull/482)

* Add support for building multiple backend simulators.
  [(#497)](https://github.com/PennyLaneAI/pennylane-lightning/pull/497)

### Documentation

### Bug fixes

* Fix CI issues running python-cov with MPI.
  [(#535)](https://github.com/PennyLaneAI/pennylane-lightning/pull/535)

* Re-add support for `pip install pennylane-lightning[gpu]`.
  [(#515)](https://github.com/PennyLaneAI/pennylane-lightning/pull/515)

* Switch most Lightning-Qubit default kernels to `LM`. Add `LM::multiQubitOp` tests, failing when targeting out-of-order wires clustered close to `num_qubits-1`. Fix the `LM::multiQubitOp` kernel implementation by introducing a generic `revWireParity` routine and replacing the `bitswap`-based implementation. Mimic the changes fixing the corresponding `multiQubitOp` and `expval` functors in Lightning-Kokkos.
  [(#511)](https://github.com/PennyLaneAI/pennylane-lightning/pull/511)

* Fix RTD builds by removing unsupported `system_packages` configuration option.
  [(#491)](https://github.com/PennyLaneAI/pennylane-lightning/pull/491)

### Contributors

This release contains contributions from (in alphabetical order):

Ali Asadi, Amintor Dusko, Vincent Michaud-Rioux, Lee J. O'Riordan, Shuli Shu

---

# Release 0.32.0

### New features since last release

* The `lightning.kokkos` backend supports Nvidia GPU execution (with Kokkos v4 and CUDA v12).
  [(#477)](https://github.com/PennyLaneAI/pennylane-lightning/pull/477)

* Complete overhaul of repository structure to facilitates integration of multiple backends. Refactoring efforts we directed to improve development performance, code reuse and decrease overall overhead to propagate changes through backends. New C++ modular build strategy allows for faster test builds restricted to a module. Update CI/CD actions concurrency strategy. Change minimal Python version to 3.9.
  [(#472)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/472)

* Wheels are built with native support for sparse Hamiltonians.
  [(#470)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/470)

* Add native support to sparse Hamiltonians in the absence of Kokkos & Kokkos-kernels.
  [(#465)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/465)

### Breaking changes

* Rename `QubitStateVector` to `StatePrep` in the Lightning-Qubit and `Lightning-Kokkos` classes.
  [(#486)](https://github.com/PennyLaneAI/pennylane-lightning/pull/486)

* Modify `adjointJacobian` methods to accept a (maybe unused) reference `StateVectorT`, allowing device-backed simulators to directly access state vector data for adjoint differentiation instead of copying it back-and-forth into `JacobianData` (host memory).
  [(#477)](https://github.com/PennyLaneAI/pennylane-lightning/pull/477)

### Improvements

* Refactor LKokkos `Measurements` class to use (fast) specialized functors whenever possible.
  [(#481)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/481)

* Merge Lightning Qubit and Lightning Kokkos backends in the new repository.
  [(#472)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/472)

* Integrated new unified docs for Lightning Kokkos and Lightning Qubit packages.
  [(#473)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/473)

### Documentation

### Bug fixes

* Ensure PennyLane has an `active_return` attribute before calling it.
 [(#483)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/483)

* Do no import `sqrt2_v` from `<numbers>` in `Util.hpp` to resolve issue with Lightning-GPU builds.
  [(#479)](https://github.com/PennyLaneAI/pennylane-lightning/pull/479)

* Update the CMake internal references to enable sub-project compilation with affecting the parent package.
  [(#478)](https://github.com/PennyLaneAI/pennylane-lightning/pull/478)

* `apply` no longer mutates the inputted list of operations.
  [(#474)](https://github.com/PennyLaneAI/pennylane-lightning/pull/474)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Christina Lee, Vincent Michaud-Rioux, Lee J. O'Riordan

---

# Release 0.31.0

### New features since last release

* Update Kokkos support to 4.0.01.
  [(#439)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/439)

### Breaking changes

* Update tests to be compliant with PennyLane v0.31.0 development changes and deprecations.
  [(#448)](https://github.com/PennyLaneAI/pennylane-lightning/pull/448)

### Improvements

* Remove logic from `setup.py` and transfer paths and env variable definitions into workflow files.
  [(#450)](https://github.com/PennyLaneAI/pennylane-lightning/pull/450)

* Detect MKL or CBLAS if `ENABLE_BLAS=ON` making sure that BLAS is linked as expected.
  [(#449)](https://github.com/PennyLaneAI/pennylane-lightning/pull/449)

### Documentation

* Fix LightningQubit class parameter documentation.
  [(#456)](https://github.com/PennyLaneAI/pennylane-lightning/pull/456)

### Bug fixes

* Ensure cross-platform wheels continue to build with updates in git safety checks.
  [(#452)](https://github.com/PennyLaneAI/pennylane-lightning/pull/452)

* Fixing Python version bug introduce in [(#450)](https://github.com/PennyLaneAI/pennylane-lightning/pull/450)
  when `Python_EXECUTABLE` was removed from `setup.py`.
  [(#461)](https://github.com/PennyLaneAI/pennylane-lightning/pull/461)

* Ensure aligned allocator definition works with C++20 compilers.
  [(#438)](https://github.com/PennyLaneAI/pennylane-lightning/pull/438)

* Prevent multiple threads from calling `Kokkos::initialize` or `Kokkos::finalize`.
  [(#439)](https://github.com/PennyLaneAI/pennylane-lightning/pull/439)

### Contributors

This release contains contributions from (in alphabetical order):

Vincent Michaud-Rioux, Lee J. O'Riordan, Chae-Yeun Park

---

# Release 0.30.0

### New features since last release

* Add MCMC sampler.
  [(#384)] (https://github.com/PennyLaneAI/pennylane-lightning/pull/384)

* Serialize PennyLane's arithmetic operators when they are used as observables
  that are expressed in the Pauli basis.
  [(#424)](https://github.com/PennyLaneAI/pennylane-lightning/pull/424)

### Breaking changes

* Lightning now works with the new return types specification that is now default in PennyLane.
  See [the PennyLane `qml.enable_return`](https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html?highlight=enable_return) documentation for more information on this change.
  [(#427)](https://github.com/PennyLaneAI/pennylane-lightning/pull/427)

Instead of creating potentially ragged numpy array, devices and `QNode`'s now return an object of the same type as that
returned by the quantum function.

```
>>> dev = qml.device('lightning.qubit', wires=1)
>>> @qml.qnode(dev, diff_method="adjoint")
... def circuit(x):
...     qml.RX(x, wires=0)
...     return qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))
>>> x = qml.numpy.array(0.5)
>>> circuit(qml.numpy.array(0.5))
(array(-0.47942554), array(0.87758256))
```

Interfaces like Jax or Torch handle tuple outputs without issues:

```
>>> jax.jacobian(circuit)(jax.numpy.array(0.5))
(Array(-0.87758255, dtype=float32, weak_type=True),
Array(-0.47942555, dtype=float32, weak_type=True))
```

Autograd cannot differentiate an output tuple, so results must be converted to an array before
use with `qml.jacobian`:

```
>>> qml.jacobian(lambda y: qml.numpy.array(circuit(y)))(x)
array([-0.87758256, -0.47942554])
```

Alternatively, the quantum function itself can return a numpy array of measurements:

```
>>> dev = qml.device('lightning.qubit', wires=1)
>>> @qml.qnode(dev, diff_method="adjoint")
>>> def circuit2(x):
...     qml.RX(x, wires=0)
...     return np.array([qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))])
>>> qml.jacobian(circuit2)(np.array(0.5))
array([-0.87758256, -0.47942554])
```

### Improvements

* Remove deprecated `set-output` commands from workflow files.
  [(#437)](https://github.com/PennyLaneAI/pennylane-lightning/pull/437)

* Lightning wheels are now checked with `twine check` post-creation for PyPI compatibility.
  [(#430)](https://github.com/PennyLaneAI/pennylane-lightning/pull/430)

* Lightning has been made compatible with the change in return types specification.
  [(#427)](https://github.com/PennyLaneAI/pennylane-lightning/pull/427)

* Lightning is compatible with clang-tidy version 16.
  [(#429)](https://github.com/PennyLaneAI/pennylane-lightning/pull/429)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Vincent Michaud-Rioux, Lee James O'Riordan, Chae-Yeun Park, Matthew Silverman

---

# Release 0.29.0

### Improvements

* Remove runtime dependency on ninja build system.
  [(#414)](https://github.com/PennyLaneAI/pennylane-lightning/pull/414)

* Allow better integration and installation support with CMake targeted binary builds.
  [(#403)](https://github.com/PennyLaneAI/pennylane-lightning/pull/403)

* Remove explicit Numpy and Scipy requirements.
  [(#412)](https://github.com/PennyLaneAI/pennylane-lightning/pull/412)

* Get `llvm` installation root from the environment variable `LLVM_ROOT_DIR` (or fallback to `brew`).
  [(#413)](https://github.com/PennyLaneAI/pennylane-lightning/pull/413)

* Update AVX2/512 kernel infrastructure for additional gate/generator operations.
  [(#404)](https://github.com/PennyLaneAI/pennylane-lightning/pull/404)

* Remove unnecessary lines for resolving CodeCov issue.
  [(#415)](https://github.com/PennyLaneAI/pennylane-lightning/pull/415)

* Add more AVX2/512 gate operations.
  [(#393)](https://github.com/PennyLaneAI/pennylane-lightning/pull/393)

### Documentation

### Bug fixes

* Ensure error raised when asking for out of order marginal probabilities. Prevents the return of incorrect results.
  [(#416)](https://github.com/PennyLaneAI/pennylane-lightning/pull/416)

* Fix Github shields in README.
  [(#402)](https://github.com/PennyLaneAI/pennylane-lightning/pull/402)

### Contributors

Amintor Dusko, Vincent Michaud-Rioux, Lee James O'Riordan, Chae-Yeun Park

---

# Release 0.28.2

### Bug fixes

* Fix Python module versioning for Linux wheels.
  [(#408)](https://github.com/PennyLaneAI/pennylane-lightning/pull/408)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Shuli Shu, Trevor Vincent

---

# Release 0.28.1

### Bug fixes

* Fix Pybind11 module versioning and locations for Windows wheels.
  [(#400)](https://github.com/PennyLaneAI/pennylane-lightning/pull/400)

### Contributors

This release contains contributions from (in alphabetical order):

Lee J. O'Riordan

---

# Release 0.28.0

### Breaking changes

* Deprecate support for Python 3.7.
  [(#391)](https://github.com/PennyLaneAI/pennylane-lightning/pull/391)

### Improvements

* Improve Lightning package structure for external use as a C++ library.
  [(#369)](https://github.com/PennyLaneAI/pennylane-lightning/pull/369)

* Improve the stopping condition method.
  [(#386)](https://github.com/PennyLaneAI/pennylane-lightning/pull/386)

### Bug fixes

- Pin CMake to 3.24.x in wheel-builder to avoid Python not found error in CMake 3.25, when building wheels for PennyLane-Lightning-GPU.
  [(#387)](https://github.com/PennyLaneAI/pennylane-lightning/pull/387)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Lee J. O'Riordan

---

# Release 0.27.0

### New features since last release

* Enable building of Python 3.11 wheels and upgrade Python on CI/CD workflows to 3.8.
  [(#381)](https://github.com/PennyLaneAI/pennylane-lightning/pull/381)

### Breaking changes

### Improvements

* Update clang-tools version in Github workflows.
  [(#351)](https://github.com/PennyLaneAI/pennylane-lightning/pull/351)

* Improve tests and checks CI/CD pipelines.
  [(#353)](https://github.com/PennyLaneAI/pennylane-lightning/pull/353)

* Implement 3 Qubits gates (CSWAP & Toffoli) & 4 Qubits gates (DoubleExcitation, DoubleExcitationMinus, DoubleExcitationPlus) in LM manner.
  [(#362)](https://github.com/PennyLaneAI/pennylane-lightning/pull/362)

* Upgrade Kokkos and Kokkos Kernels to 3.7.00, and improve sparse matrix-vector multiplication performance and memory usage.
  [(#361)](https://github.com/PennyLaneAI/pennylane-lightning/pull/361)

* Update Linux (ubuntu-latest) architecture x86_64 wheel-builder from GCC 10.x to GCC 11.x.
  [(#373)](https://github.com/PennyLaneAI/pennylane-lightning/pull/373)

* Update gcc and g++ 10.x to 11.x in CI tests. This update brings improved support for newer C++ features.
  [(#370)](https://github.com/PennyLaneAI/pennylane-lightning/pull/370)

* Change Lightning to inherit from QubitDevice instead of DefaultQubit.
  [(#365)](https://github.com/PennyLaneAI/pennylane-lightning/pull/365)

### Documentation

### Bug fixes

* Use mutex when accessing cache in KernelMap.
  [(#382)](https://github.com/PennyLaneAI/pennylane-lightning/pull/382)

### Contributors

This release contains contributions from (in alphabetical order):

Amintor Dusko, Chae-Yeun Park, Monit Sharma, Shuli Shu

---

# Release 0.26.1

### Bug fixes

* Fixes the transposition method used in the probability calculation.
  [(#377)](https://github.com/PennyLaneAI/pennylane-lightning/pull/377)

### Contributor

Amintor Dusko

---
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

* Fixes a bug where the `QNode` would swap Lightning-Qubit to
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
