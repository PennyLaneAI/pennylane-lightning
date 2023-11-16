// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file MeasurementsBase.hpp
 * Defines the Measurements CRTP base class.
 */
#pragma once

#include <string>
#include <vector>

#include "Observables.hpp"

#include "CPUMemoryModel.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Observables;
} // namespace
/// @endcond

namespace Pennylane::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix), the observable
 * class, or through a string-based function dispatch.
 *
 * @tparam StateVectorT
 * @tparam Derived
 */
template <class StateVectorT, class Derived> class MeasurementsBase {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

  protected:
#ifdef _ENABLE_PLGPU
    StateVectorT &_statevector;
#else
    const StateVectorT &_statevector;
#endif

  public:
#ifdef _ENABLE_PLGPU
    explicit MeasurementsBase(StateVectorT &statevector)
        : _statevector{statevector} {};
#else
    explicit MeasurementsBase(const StateVectorT &statevector)
        : _statevector{statevector} {};
#endif

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->expval(obs);
    }

    /**
     * @brief Calculate the variance for a general Observable.
     *
     * @param obs Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        return static_cast<Derived *>(this)->var(obs);
    }

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs();
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        return static_cast<Derived *>(this)->probs(wires);
    };

    /**
     * @brief  Generate samples
     *
     * @param num_samples Number of samples
     * @return 1-D vector of samples in binary with each sample
     * separated by a stride equal to the number of qubits.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {
        return static_cast<Derived *>(this)->generate_samples(num_samples);
    };

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     *
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range = {}) -> PrecisionT {
        PrecisionT result = 0;

        if (obs.getObsName().find("SparseHamiltonian") != std::string::npos) {
            PL_ABORT("For SparseHamiltonian Observables, expval calculation is "
                     "not supported by shots");
        } else if (obs.getObsName().find("Hermitian") != std::string::npos) {
            PL_ABORT("For Hermitian Observables, expval calculation is not "
                     "supported by shots");
        } else if (obs.getObsName().find("Hamiltonian") != std::string::npos) {
            auto coeffs = obs.getCoeffs();
            for (size_t obs_term_idx = 0; obs_term_idx < coeffs.size();
                 obs_term_idx++) {
                auto obs_samples = measure_with_samples(
                    obs, num_shots, shot_range, obs_term_idx);
                PrecisionT result_per_term = std::accumulate(
                    obs_samples.begin(), obs_samples.end(), 0.0);

                result +=
                    coeffs[obs_term_idx] * result_per_term / obs_samples.size();
            }
        } else {
            auto obs_samples = measure_with_samples(obs, num_shots, shot_range);
            result =
                std::accumulate(obs_samples.begin(), obs_samples.end(), 0.0);
            result = result / obs_samples.size();
        }
        return result;
    }

    /**
     * @brief Calculate the expectation value for a general Observable.
     *
     * @param obs Observable.
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used
     * by default.
     * @param term_idx Index of a Hamiltonian term
     *
     * @return Expectation value with respect to the given observable.
     */
    auto measure_with_samples(const Observable<StateVectorT> &obs,
                              const size_t &num_shots,
                              const std::vector<size_t> &shot_range,
                              const size_t term_idx = 0) {
        const size_t num_qubits = _statevector.getTotalNumQubits();
        std::vector<size_t> obs_wires;
        std::vector<size_t> identity_wires;

        auto sub_samples = _sample_state(obs, num_shots, shot_range, obs_wires,
                                         identity_wires, term_idx);

        std::vector<PrecisionT> obs_samples(num_shots, 0);

        size_t num_identity_obs = identity_wires.size();
        if (!identity_wires.empty()) {
            size_t identity_obs_idx = 0;
            for (size_t i = 0; i < obs_wires.size(); i++) {
                if (identity_wires[identity_obs_idx] == obs_wires[i]) {
                    std::swap(obs_wires[identity_obs_idx], obs_wires[i]);
                    identity_obs_idx++;
                }
            }
        }

        for (size_t i = 0; i < num_shots; i++) {
            std::vector<size_t> local_sample;
            for (auto &obs_wire : obs_wires) {
                local_sample.push_back(sub_samples[i * num_qubits + obs_wire]);
            }

            if (num_identity_obs != obs_wires.size()) {
                // eigen values are `1` and `-1` for PauliX, PauliY, PauliZ,
                // Hadamard gates the eigen value for a eigen vector |00001> is
                // -1 since sum of the value at each bit position is odd
                if ((static_cast<size_t>(std::accumulate(
                         local_sample.begin() + num_identity_obs,
                         local_sample.end(), 0)) &
                     size_t{1}) == 1) {
                    obs_samples[i] = -1;
                } else {
                    obs_samples[i] = 1;
                }
            } else {
                // eigen value for Identity gate is `1`
                obs_samples[i] = 1;
            }
        }
        return obs_samples;
    }

  private:
    /**
     * @brief Return preprocess state with a observable
     *
     * @param obs The observable to sample
     * @param obs_wires Observable wires.
     * @param identity_wires Wires of Identity gates
     * @param term_idx Index of a Hamiltonian term
     *
     * @return a StateVectorT object
     */
    auto _preprocess_state(const Observable<StateVectorT> &obs,
                           std::vector<size_t> &obs_wires,
                           std::vector<size_t> &identity_wires,
                           const size_t &term_idx = 0) {
        if constexpr (std::is_same_v<
                          typename StateVectorT::MemoryStorageT,
                          Pennylane::Util::MemoryStorageLocation::External>) {
            StateVectorT sv(_statevector.getData(), _statevector.getLength());
            sv.updateData(_statevector.getData(), _statevector.getLength());
            obs.applyInPlaceShots(sv, identity_wires, obs_wires, term_idx);
            return sv;
        } else {
            StateVectorT sv(_statevector);
            obs.applyInPlaceShots(sv, identity_wires, obs_wires, term_idx);
            return sv;
        }
    }

    /**
     * @brief Return samples of a observable
     *
     * @param obs The observable to sample
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. All samples are used by
     * default.
     * @param obs_wires Observable wires.
     * @param identity_wires Wires of Identity gates
     * @param term_idx Index of a Hamiltonian term
     *
     * @return std::vector<size_t> samples in std::vector
     */
    auto _sample_state(const Observable<StateVectorT> &obs,
                       const size_t &num_shots,
                       const std::vector<size_t> &shot_range,
                       std::vector<size_t> &obs_wires,
                       std::vector<size_t> &identity_wires,
                       const size_t &term_idx = 0) {
        const size_t num_qubits = _statevector.getTotalNumQubits();
        auto sv = _preprocess_state(obs, obs_wires, identity_wires, term_idx);
        Derived measure(sv);
        auto samples = measure.generate_samples(num_shots);

        if (!shot_range.empty()) {
            std::vector<size_t> sub_samples(shot_range.size() * num_qubits);
            // Get a slice of samples based on the shot_range vector
            size_t shot_idx = 0;
            for (const auto &i : shot_range) {
                for (size_t j = i * num_qubits; j < (i + 1) * num_qubits; j++) {
                    // TODO some extra work to make it cache-friendly
                    sub_samples[shot_idx * num_qubits + j - i * num_qubits] =
                        samples[j];
                }
                shot_idx++;
            }
            return sub_samples;
        }
        return samples;
    }
};
} // namespace Pennylane::Measures
