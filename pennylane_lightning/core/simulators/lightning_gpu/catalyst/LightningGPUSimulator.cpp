// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <unordered_set>

#include "LightningGPUSimulator.hpp"

namespace Catalyst::Runtime::Simulator {

auto LightningGPUSimulator::AllocateQubit() -> QubitIdType {
    const size_t num_qubits = GetNumQubits();
    if (num_qubits == 0U) {
        this->device_sv = std::make_unique<StateVectorT>(1);
        return this->qubit_manager.Allocate(0);
    }

    // The statevector may contain previously freed qubits,
    // that means we may not need to resize the vector.
    size_t device_idx;
    QubitIdType new_program_idx;
    std::optional<size_t> candidate = this->qubit_manager.popFreeQubit();
    if (!candidate.has_value()) {
        // TODO: update statevector directly on device
        const auto &original_data = this->device_sv->getDataVector();

        const size_t dsize = original_data.size();
        RT_ASSERT(dsize == 1UL << num_qubits);

        std::vector<std::complex<double>> new_data(dsize << 1UL);

        device_idx = num_qubits;
        for (size_t i = 0; i < original_data.size(); i++) {
            new_data[2 * i] = original_data[i];
        }
        this->device_sv =
            std::make_unique<StateVectorT>(new_data.data(), new_data.size());
        new_program_idx = this->qubit_manager.Allocate(device_idx);
    } else {
        device_idx = candidate.value();

        // Reuse existing space in the statevector by collapsing onto |0>.
        // The collapse is performed by a measurement followed by an X gate if
        // measured 1.
        new_program_idx = this->qubit_manager.Allocate(device_idx);
        Result mres = this->Measure(new_program_idx);
        if (*mres) {
            this->NamedOperation("PauliX", {}, {new_program_idx}, false, {},
                                 {});
        }
    }

    return new_program_idx;
}

auto LightningGPUSimulator::AllocateQubits(std::size_t num_qubits)
    -> std::vector<QubitIdType> {
    if (!num_qubits) {
        return {};
    }

    // at the first call when num_qubits == 0
    if (!this->GetNumQubits()) {
        this->device_sv = std::make_unique<StateVectorT>(num_qubits);
        return this->qubit_manager.AllocateRange(0, num_qubits);
    }

    std::vector<QubitIdType> result(num_qubits);
    std::generate_n(result.begin(), num_qubits,
                    [this]() { return AllocateQubit(); });
    return result;
}

void LightningGPUSimulator::ReleaseQubit(QubitIdType q) {
    // We do not deallocate physical memory in the statevector for this
    // operation, instead we just mark the qubits as released.
    this->qubit_manager.Release(q);
}

void LightningGPUSimulator::ReleaseQubits(const std::vector<QubitIdType> &ids) {
    // fast path for single register alloc and dealloc
    if (GetNumQubits() == ids.size()) {
        std::vector<QubitIdType> allocated_ids =
            this->qubit_manager.getAllQubitIds();
        std::unordered_set<QubitIdType> allocated_set(allocated_ids.begin(),
                                                      allocated_ids.end());
        bool deallocate_all =
            std::all_of(ids.begin(), ids.end(), [&](QubitIdType id) {
                return allocated_set.contains(id);
            });
        if (deallocate_all) {
            this->qubit_manager.ReleaseAll();
            this->device_sv = std::make_unique<StateVectorT>(0);
            return;
        }
    }

    for (auto id : ids) {
        this->qubit_manager.Release(id);
    }
}

auto LightningGPUSimulator::GetNumQubits() const -> std::size_t {
    return this->qubit_manager.getNumQubits();
}

void LightningGPUSimulator::StartTapeRecording() {
    RT_FAIL_IF(this->tape_recording, "Cannot re-activate the cache manager");
    this->tape_recording = true;
    this->cache_manager.Reset();
}

void LightningGPUSimulator::StopTapeRecording() {
    RT_FAIL_IF(!this->tape_recording,
               "Cannot stop an already stopped cache manager");
    this->tape_recording = false;
}

auto LightningGPUSimulator::CacheManagerInfo()
    -> std::tuple<std::size_t, std::size_t, std::size_t,
                  std::vector<std::string>, std::vector<ObsIdType>> {
    return {this->cache_manager.getNumOperations(),
            this->cache_manager.getNumObservables(),
            this->cache_manager.getNumParams(),
            this->cache_manager.getOperationsNames(),
            this->cache_manager.getObservablesKeys()};
}

void LightningGPUSimulator::SetDeviceShots(std::size_t shots) {
    this->device_shots = shots;
}

auto LightningGPUSimulator::GetDeviceShots() const -> std::size_t {
    return this->device_shots;
}

void LightningGPUSimulator::SetDevicePRNG(std::mt19937 *gen) {
    this->gen = gen;
}

void LightningGPUSimulator::SetState(DataView<std::complex<double>, 1> &data,
                                     std::vector<QubitIdType> &wires) {
    std::size_t expected_wires = static_cast<std::size_t>(log2(data.size()));
    RT_ASSERT(expected_wires == wires.size());
    std::vector<std::complex<double>> data_vector(data.begin(), data.end());
    this->device_sv->setStateVector(data_vector.data(), data_vector.size(),
                                    getDeviceWires(wires));
}

void LightningGPUSimulator::SetBasisState(DataView<int8_t, 1> &data,
                                          std::vector<QubitIdType> &wires) {
    std::vector<std::size_t> basis_state(data.begin(), data.end());
    this->device_sv->setBasisState(basis_state, getDeviceWires(wires));
}

void LightningGPUSimulator::NamedOperation(
    const std::string &name, const std::vector<double> &params,
    const std::vector<QubitIdType> &wires, bool inverse,
    const std::vector<QubitIdType> &controlled_wires,
    const std::vector<bool> &controlled_values) {
    // Check the validity of number of qubits and parameters
    RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
               "Controlled wires/values size mismatch");
    RT_FAIL_IF(!isValidQubits(wires), "Given wires do not refer to qubits");
    RT_FAIL_IF(!isValidQubits(controlled_wires),
               "Given controlled wires do not refer to qubits");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);
    auto &&dev_controlled_wires = getDeviceWires(controlled_wires);

    // Update the state-vector
    if (controlled_wires.empty()) {
        this->device_sv->applyOperation(name, dev_wires, inverse, params);
    } else {
        this->device_sv->applyOperation(name, dev_controlled_wires,
                                        controlled_values, dev_wires, inverse,
                                        params);
    }

    // Update tape caching if required
    if (this->tape_recording) {
        this->cache_manager.addOperation(name, params, dev_wires, inverse, {},
                                         dev_controlled_wires,
                                         controlled_values);
    }
}

void LightningGPUSimulator::MatrixOperation(
    const std::vector<std::complex<double>> &matrix,
    const std::vector<QubitIdType> &wires, bool inverse,
    const std::vector<QubitIdType> &controlled_wires,
    const std::vector<bool> &controlled_values) {
    RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
               "Controlled wires/values size mismatch");
    RT_FAIL_IF(!isValidQubits(wires), "Given wires do not refer to qubits");
    RT_FAIL_IF(!isValidQubits(controlled_wires),
               "Given controlled wires do not refer to qubits");

    // Convert wires to device wires
    auto &&dev_wires = getDeviceWires(wires);
    auto &&dev_controlled_wires = getDeviceWires(controlled_wires);

    if (controlled_wires.empty()) {
        this->device_sv->applyMatrix(matrix, dev_wires, inverse);
    } else {
        this->device_sv->applyOperation("matrix", dev_controlled_wires,
                                        controlled_values, dev_wires, inverse,
                                        {}, matrix);
    }

    // Update tape caching if required
    if (this->tape_recording) {
        this->cache_manager.addOperation("QubitUnitary", {}, dev_wires, inverse,
                                         matrix, dev_controlled_wires,
                                         controlled_values);
    }
}

auto LightningGPUSimulator::Observable(
    ObsId id, const std::vector<std::complex<double>> &matrix,
    const std::vector<QubitIdType> &wires) -> ObsIdType {
    RT_FAIL_IF(wires.size() > this->GetNumQubits(), "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires");

    auto &&dev_wires = getDeviceWires(wires);

    if (id == ObsId::Hermitian) {
        return this->obs_manager.createHermitianObs(matrix, dev_wires);
    }

    return this->obs_manager.createNamedObs(id, dev_wires);
}

auto LightningGPUSimulator::TensorObservable(const std::vector<ObsIdType> &obs)
    -> ObsIdType {
    return this->obs_manager.createTensorProdObs(obs);
}

auto LightningGPUSimulator::HamiltonianObservable(
    const std::vector<double> &coeffs, const std::vector<ObsIdType> &obs)
    -> ObsIdType {
    return this->obs_manager.createHamiltonianObs(coeffs, obs);
}

auto LightningGPUSimulator::Expval(ObsIdType obsKey) -> double {
    RT_FAIL_IF(!this->obs_manager.isValidObservables({obsKey}),
               "Invalid key for cached observables");

    // update tape caching
    if (this->tape_recording) {
        cache_manager.addObservable(obsKey, MeasurementsT::Expval);
    }

    auto &&obs = this->obs_manager.getObservable(obsKey);

    Pennylane::LightningGPU::Measures::Measurements<StateVectorT> m{
        *(this->device_sv)};

    m.setSeed(this->generateSeed());

    return device_shots ? m.expval(*obs, device_shots, {}) : m.expval(*obs);
}

auto LightningGPUSimulator::Var(ObsIdType obsKey) -> double {
    RT_FAIL_IF(!this->obs_manager.isValidObservables({obsKey}),
               "Invalid key for cached observables");

    // update tape caching
    if (this->tape_recording) {
        this->cache_manager.addObservable(obsKey, MeasurementsT::Var);
    }

    auto &&obs = this->obs_manager.getObservable(obsKey);

    Pennylane::LightningGPU::Measures::Measurements<StateVectorT> m{
        *(this->device_sv)};

    m.setSeed(this->generateSeed());

    return device_shots ? m.var(*obs, device_shots) : m.var(*obs);
}

void LightningGPUSimulator::State(DataView<std::complex<double>, 1> &state) {
    const std::size_t num_qubits = this->device_sv->getNumQubits();
    const std::size_t size = Pennylane::Util::exp2(num_qubits);
    RT_FAIL_IF(state.size() != size,
               "Invalid size for the pre-allocated state vector");

    // create a temporary buffer to copy the underlying state-vector to
    std::vector<std::complex<double>> buffer(size);
    // copy data from device to host
    this->device_sv->CopyGpuDataToHost(buffer.data(), size);

    // move data to state leveraging MemRefIter
    std::move(buffer.begin(), buffer.end(), state.begin());
}

void LightningGPUSimulator::Probs(DataView<double, 1> &probs) {
    Pennylane::LightningGPU::Measures::Measurements<StateVectorT> m{
        *(this->device_sv)};

    m.setSeed(this->generateSeed());

    auto &&dv_probs = device_shots ? m.probs(device_shots) : m.probs();

    RT_FAIL_IF(probs.size() != dv_probs.size(),
               "Invalid size for the pre-allocated probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

void LightningGPUSimulator::PartialProbs(
    DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) {
    const std::size_t numWires = wires.size();
    const std::size_t numQubits = this->GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");

    auto dev_wires = getDeviceWires(wires);
    Pennylane::LightningGPU::Measures::Measurements<StateVectorT> m{
        *(this->device_sv)};

    m.setSeed(this->generateSeed());

    auto &&dv_probs =
        device_shots ? m.probs(dev_wires, device_shots) : m.probs(dev_wires);

    RT_FAIL_IF(probs.size() != dv_probs.size(),
               "Invalid size for the pre-allocated partial-probabilities");

    std::move(dv_probs.begin(), dv_probs.end(), probs.begin());
}

std::vector<size_t> LightningGPUSimulator::GenerateSamples(size_t shots) {
    // generate_samples is a member function of the Measures class.
    Pennylane::LightningGPU::Measures::Measurements<StateVectorT> m{
        *(this->device_sv)};

    m.setSeed(this->generateSeed());

    return m.generate_samples(shots);
}

void LightningGPUSimulator::Sample(DataView<double, 2> &samples) {
    auto li_samples = this->GenerateSamples(device_shots);

    RT_FAIL_IF(samples.size() != li_samples.size(),
               "Invalid size for the pre-allocated samples");

    const std::size_t numQubits = this->GetNumQubits();

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    auto samplesIter = samples.begin();
    for (std::size_t shot = 0; shot < device_shots; shot++) {
        for (std::size_t wire = 0; wire < numQubits; wire++) {
            *(samplesIter++) =
                static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}
void LightningGPUSimulator::PartialSample(
    DataView<double, 2> &samples, const std::vector<QubitIdType> &wires) {
    const std::size_t numWires = wires.size();
    const std::size_t numQubits = this->GetNumQubits();

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF(samples.size() != device_shots * numWires,
               "Invalid size for the pre-allocated partial-samples");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    auto li_samples = this->GenerateSamples(device_shots);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    auto samplesIter = samples.begin();
    for (std::size_t shot = 0; shot < device_shots; shot++) {
        for (auto wire : dev_wires) {
            *(samplesIter++) =
                static_cast<double>(li_samples[shot * numQubits + wire]);
        }
    }
}

void LightningGPUSimulator::Counts(DataView<double, 1> &eigvals,
                                   DataView<int64_t, 1> &counts) {
    const std::size_t numQubits = this->GetNumQubits();
    const std::size_t numElements = 1U << numQubits;

    RT_FAIL_IF(eigvals.size() != numElements || counts.size() != numElements,
               "Invalid size for the pre-allocated counts");

    auto li_samples = this->GenerateSamples(device_shots);

    // Fill the eigenvalues with the integer representation of the
    // corresponding computational basis bitstring. In the future,
    // eigenvalues can also be obtained from an observable, hence the
    // bitstring integer is stored as a double.
    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the bits of all qubits
    // into a bitstring.
    for (std::size_t shot = 0; shot < device_shots; shot++) {
        std::bitset<CHAR_BIT * sizeof(double)> basisState;
        std::size_t idx = numQubits;
        for (std::size_t wire = 0; wire < numQubits; wire++) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<std::size_t>(basisState.to_ulong())) += 1;
    }
}

void LightningGPUSimulator::PartialCounts(
    DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
    const std::vector<QubitIdType> &wires) {
    const std::size_t numWires = wires.size();
    const std::size_t numQubits = this->GetNumQubits();
    const std::size_t numElements = 1U << numWires;

    RT_FAIL_IF(numWires > numQubits, "Invalid number of wires");
    RT_FAIL_IF(!isValidQubits(wires), "Invalid given wires to measure");
    RT_FAIL_IF((eigvals.size() != numElements || counts.size() != numElements),
               "Invalid size for the pre-allocated partial-counts");

    // get device wires
    auto &&dev_wires = getDeviceWires(wires);

    auto li_samples = this->GenerateSamples(device_shots);

    // Fill the eigenvalues with the integer representation of the
    // corresponding computational basis bitstring. In the future,
    // eigenvalues can also be obtained from an observable, hence the
    // bitstring integer is stored as a double.
    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);

    // The lightning samples are layed out as a single vector of size
    // shots*qubits, where each element represents a single bit. The
    // corresponding shape is (shots, qubits). Gather the desired bits
    // corresponding to the input wires into a bitstring.
    for (std::size_t shot = 0; shot < device_shots; shot++) {
        std::bitset<CHAR_BIT * sizeof(double)> basisState;
        std::size_t idx = dev_wires.size();
        for (auto wire : dev_wires) {
            basisState[--idx] = li_samples[shot * numQubits + wire];
        }
        counts(static_cast<std::size_t>(basisState.to_ulong())) += 1;
    }
}

auto LightningGPUSimulator::Measure(QubitIdType wire,
                                    std::optional<int32_t> postselect)
    -> Result {
    // get a measurement
    std::vector<QubitIdType> wires = {reinterpret_cast<QubitIdType>(wire)};

    std::vector<double> probs(1U << wires.size());
    DataView<double, 1> buffer_view(probs);
    auto device_shots = GetDeviceShots();
    SetDeviceShots(0);
    PartialProbs(buffer_view, wires);
    SetDeviceShots(device_shots);

    // It represents the measured result, true for 1, false for 0
    bool mres = Lightning::simulateDraw(probs, postselect, this->gen);
    auto dev_wires = getDeviceWires(wires);
    this->device_sv->collapse(dev_wires[0], mres ? 1 : 0);
    return mres ? const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST)
                : const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

void LightningGPUSimulator::Gradient(
    std::vector<DataView<double, 1>> &gradients,
    const std::vector<std::size_t> &trainParams) {
    const bool tp_empty = trainParams.empty();
    const std::size_t num_observables = this->cache_manager.getNumObservables();
    const std::size_t num_params = this->cache_manager.getNumParams();
    const std::size_t num_train_params =
        tp_empty ? num_params : trainParams.size();
    const std::size_t jac_size =
        num_train_params * this->cache_manager.getNumObservables();

    if (!jac_size) {
        return;
    }

    RT_FAIL_IF(gradients.size() != num_observables,
               "Invalid number of pre-allocated gradients");

    auto &&obs_callees = this->cache_manager.getObservablesCallees();
    bool is_valid_measurements =
        std::all_of(obs_callees.begin(), obs_callees.end(),
                    [](const auto &m) { return m == MeasurementsT::Expval; });
    RT_FAIL_IF(!is_valid_measurements,
               "Unsupported measurements to compute gradient; "
               "Adjoint differentiation method only supports expectation "
               "return type");

    // Create OpsData
    auto &&ops_names = this->cache_manager.getOperationsNames();
    auto &&ops_params = this->cache_manager.getOperationsParameters();
    auto &&ops_wires = this->cache_manager.getOperationsWires();
    auto &&ops_inverses = this->cache_manager.getOperationsInverses();
    auto &&ops_matrices = this->cache_manager.getOperationsMatrices();
    auto &&ops_controlled_wires =
        this->cache_manager.getOperationsControlledWires();
    auto &&ops_controlled_values =
        this->cache_manager.getOperationsControlledValues();

    const auto &&ops = Pennylane::Algorithms::OpsData<StateVectorT>(
        ops_names, ops_params, ops_wires, ops_inverses, ops_matrices,
        ops_controlled_wires, ops_controlled_values);

    // Create the vector of observables
    auto &&obs_keys = this->cache_manager.getObservablesKeys();
    std::vector<
        std::shared_ptr<Pennylane::Observables::Observable<StateVectorT>>>
        obs_vec;
    obs_vec.reserve(obs_keys.size());
    for (auto idx : obs_keys) {
        obs_vec.emplace_back(this->obs_manager.getObservable(idx));
    }

    std::vector<std::size_t> all_params;
    if (tp_empty) {
        all_params.reserve(num_params);
        for (std::size_t i = 0; i < num_params; i++) {
            all_params.push_back(i);
        }
    }

    // construct the Jacobian data
    Pennylane::Algorithms::JacobianData<StateVectorT> tape{
        num_params,
        this->device_sv->getLength(),
        this->device_sv->getData(),
        obs_vec,
        ops,
        tp_empty ? all_params : trainParams};

    Pennylane::LightningGPU::Algorithms::AdjointJacobian<StateVectorT> adj;
    std::vector<double> jacobian(jac_size, 0);
    adj.adjointJacobian(std::span{jacobian}, tape,
                        /* ref_data */ *this->device_sv,
                        /* apply_operations */ false);

    std::vector<double> cur_buffer(num_train_params);
    auto begin_loc_iter = jacobian.begin();
    for (std::size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
        RT_ASSERT(begin_loc_iter != jacobian.end());
        RT_ASSERT(num_train_params <= gradients[obs_idx].size());
        std::move(begin_loc_iter, begin_loc_iter + num_train_params,
                  cur_buffer.begin());
        std::move(cur_buffer.begin(), cur_buffer.end(),
                  gradients[obs_idx].begin());
        begin_loc_iter += num_train_params;
    }
}

} // namespace Catalyst::Runtime::Simulator

/// LCOV_EXCL_START
GENERATE_DEVICE_FACTORY(LightningGPUSimulator,
                        Catalyst::Runtime::Simulator::LightningGPUSimulator);
/// LCOV_EXCL_STOP
