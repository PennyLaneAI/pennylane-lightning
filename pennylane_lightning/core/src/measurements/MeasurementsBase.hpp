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
        PrecisionT result{0.0};

        if (obs.getObsName().find("SparseHamiltonian") != std::string::npos) {
            // SparseHamiltonian does not support samples in pennylane.
            PL_ABORT("For SparseHamiltonian Observables, expval calculation is "
                     "not supported by shots");
        } else if (obs.getObsName().find("Hermitian") != std::string::npos) {
            // TODO support. This support requires an additional method to solve
            // eigenpair and unitary matrices, and the results of eigenpair and
            // unitary matrices data need to be added to the Hermitian class and
            // public methods are need to access eigen values. Note the
            // assumption that eigen values are -1 and 1 in the
            // `measurement_with_sample` method should be updated as well.
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
            result /= obs_samples.size();
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
                              size_t term_idx = 0) {
        const size_t num_qubits = _statevector.getTotalNumQubits();
        std::vector<size_t> obs_wires;
        std::vector<size_t> identity_wires;

        auto sub_samples = _sample_state(obs, num_shots, shot_range, obs_wires,
                                         identity_wires, term_idx);

        size_t num_samples = shot_range.empty() ? num_shots : shot_range.size();

        std::vector<PrecisionT> obs_samples(num_samples, 0);

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

        for (size_t i = 0; i < num_samples; i++) {
            std::vector<size_t> local_sample(obs_wires.size());
            size_t idx = 0;
            for (auto &obs_wire : obs_wires) {
                local_sample[idx] = sub_samples[i * num_qubits + obs_wire];
                idx++;
            }

            if (num_identity_obs != obs_wires.size()) {
                // eigen values are `1` and `-1` for PauliX, PauliY, PauliZ,
                // Hadamard gates the eigen value for a eigen vector |00001> is
                // -1 since sum of the value at each bit position is odd
                size_t bitSum = static_cast<size_t>(
                    std::accumulate(local_sample.begin() + num_identity_obs,
                                    local_sample.end(), 0));
                if ((bitSum & size_t{1}) == 1) {
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

    /**
     * @brief Calculate the variance for an observable with the number of shots.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots used to generate samples
     *
     * @return Variance of the given observable.
     */
    auto var(const Observable<StateVectorT> &obs, const size_t &num_shots) {
        if (obs.getObsName().find("Hamiltonian") == std::string::npos) {
            // Branch for NamedObs and TensorProd observables
            auto square_mean = expval(obs, num_shots, {});
            PrecisionT result =
                1 - square_mean *
                        square_mean; //`1` used here is because Eigenvalues for
                                     // Paulis, Hadamard and Identity are {-1,
                                     // 1}. Need to change based on eigen values
                                     // when add Hermitian support.
            return result;
        }
        // Branch for Hamiltonian observables
        auto coeffs = obs.getCoeffs();
        PrecisionT result{0.0};
        size_t obs_term_idx = 0;
        for (const auto &coeff : coeffs) {
            std::vector<size_t> shot_range = {};
            auto obs_samples =
                measure_with_samples(obs, num_shots, shot_range, obs_term_idx);
            PrecisionT expval_per_term =
                std::accumulate(obs_samples.begin(), obs_samples.end(), 0.0);
            auto term_mean = expval_per_term / obs_samples.size();

            result +=
                coeff * coeff *
                (1 - term_mean *
                         term_mean); //`1` used here is because Eigenvalues for
                                     // Paulis, Hadamard and Identity are {-1,
                                     // 1}. Need to change based on eigen values
                                     // when add Hermitian support.
            obs_term_idx++;
        }
        return result;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots (Optional). If specified with a non-zero
     * number, shot-noise will be added to return probabilities
     *
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto probs(const Observable<StateVectorT> &obs,
               const size_t &num_shots = 0) {
        PL_ABORT_IF(
            obs.getObsName().find("Hamiltonian") != std::string::npos,
            "Hamiltonian and Sparse Hamiltonian do not support samples().");
        std::vector<size_t> obs_wires;
        std::vector<size_t> identity_wires;
        auto sv = _preprocess_state(obs, obs_wires, identity_wires);
        Derived measure(sv);
        if (num_shots!=size_t{0}) {
            return measure.probs(obs_wires, num_shots);
        }
        return measure.probs(obs_wires);
    }

    /**
     * @brief Probabilities with shot-noise for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(const std::vector<size_t> &wires, const size_t &num_shots)
        -> std::vector<PrecisionT> {
        auto counts_map = counts(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();

        std::vector<PrecisionT> prob_shots(size_t{1} << wires.size(), 0.0);

        for (auto &it : counts_map) {
            size_t bitVal = 0;
            for (size_t bit = 0; bit < wires.size(); bit++) {
                // Mapping the value of wires[bit]th bit to local [bit]th bit of
                // the output
                bitVal += ((it.first >> (num_wires - size_t{1} - wires[bit])) &
                           size_t{1})
                          << (wires.size() - size_t{1} - bit);
            }

            prob_shots[bitVal] +=
                it.second / static_cast<PrecisionT>(num_shots);
        }

        return prob_shots;
    }

    /**
     * @brief Probabilities with shot-noise.
     *
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(const size_t &num_shots) -> std::vector<PrecisionT> {
        auto counts_map = counts(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();

        std::vector<PrecisionT> prob_shots(size_t{1} << num_wires, 0.0);

        for (auto &it : counts_map) {
            prob_shots[it.first] =
                it.second / static_cast<PrecisionT>(num_shots);
        }

        return prob_shots;
    }

    /**
     * @brief Return samples drawn from eigenvalues of the observable
     *
     * @param obs The observable object to sample
     * @param num_shots Number of shots used to generate samples
     *
     * @return Samples of eigenvalues of the observable
     */
    auto sample(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            obs.getObsName().find("Hamiltonian") != std::string::npos,
            "Hamiltonian and Sparse Hamiltonian do not support samples().");
        std::vector<size_t> obs_wires;
        std::vector<size_t> identity_wires;
        std::vector<size_t> shot_range = {};
        size_t term_idx = 0;

        return measure_with_samples(obs, num_shots, shot_range, term_idx);
    }

    /**
     * @brief Return the raw basis state samples
     *
     * @param num_shots Number of shots used to generate samples
     *
     * @return Raw basis state samples
     */
    auto sample(const size_t &num_shots) -> std::vector<size_t> {
        Derived measure(_statevector);
        return measure.generate_samples(num_shots);
    }

    /**
     * @brief Groups the eigenvalues of samples into a dictionary showing
     * number of occurences for each possible outcome with the number of shots.
     *
     * @param obs The observable to sample
     * @param num_shots Number of wires the sampled observable was performed on
     *
     * @return std::unordered_map<PrecisionT, size_t> with format
     * ``{'EigenValue': num_occurences}``
     */
    auto counts(const Observable<StateVectorT> &obs, const size_t &num_shots)
        -> std::unordered_map<PrecisionT, size_t> {
        std::unordered_map<PrecisionT, size_t> outcome_map;
        auto sample_data = sample(obs, num_shots);
        for (size_t i = 0; i < num_shots; i++) {
            auto key = sample_data[i];
            auto it = outcome_map.find(key);
            if (it != outcome_map.end()) {
                it->second += 1;
            } else {
                outcome_map[key] = 1;
            }
        }
        return outcome_map;
    }

    /**
     * @brief Groups the samples into a dictionary showing number of occurences
     * for each possible outcome with the number of shots.
     *
     * @param num_shots Number of wires the sampled observable was performed on
     *
     * @return std::unordered_map<size_t, size_t> with format ``{'outcome':
     * num_occurences}``
     */
    auto counts(const size_t &num_shots) -> std::unordered_map<size_t, size_t> {
        std::unordered_map<size_t, size_t> outcome_map;
        auto sample_data = sample(num_shots);

        size_t num_wires = _statevector.getTotalNumQubits();
        for (size_t i = 0; i < num_shots; i++) {
            size_t key = 0;
            for (size_t j = 0; j < num_wires; j++) {
                key += sample_data[i * num_wires + j] << (num_wires - 1 - j);
            }

            auto it = outcome_map.find(key);
            if (it != outcome_map.end()) {
                it->second += 1;
            } else {
                outcome_map[key] = 1;
            }
        }
        return outcome_map;
    }

  private:
    /**
     * @brief Return preprocess state with a observable
     *
     * @param obs The observable to sample
     * @param obs_wires Observable wires.
     * @param identity_wires Wires of Identity gates
     * @param term_idx Index of a Hamiltonian term. For other observables, its
     * value is 0, which is set as default.
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
     * @param term_idx Index of a Hamiltonian term. For other observables, its
     * value is 0, which is set as default.
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
