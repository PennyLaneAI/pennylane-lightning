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

#include <vector>

#include "Observables.hpp"

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
     * @param shots Vector of shot number to measurement
     * @param shot_range The range of samples to use. If it's empty, all samples
     * are used.
     *
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs, size_t &num_shots,
                std::vector<size_t> &shot_range) -> PrecisionT {
        PrecisionT result = 0;
        std::vector<size_t> short_range = {};
        auto obs_samples = samples(obs, num_shots, shot_range);

        size_t num_elements = 0;
        for (int element : obs_samples) {
            result += element;
            num_elements++;
        }

        return result / num_elements;
    }

    /**
     * @brief Return samples of a observable
     *
     * @param obs The observable to sample
     * @param num_shots Number of shots used to generate samples
     * @param shot_range The range of samples to use. If it's empty, all samples
     * are used.
     * @param bin_size  Divides the shot range into bins of size ``bin_size``,
     * and returns the measurement statistic separately over each bin.
     * @param counts Whether count("True") or raw samples ("False") should be
     * retruned
     *
     * @return std::vector<size_t> samples in std::vector
     */
    auto samples(const Observable<StateVectorT> &obs, const size_t &num_shots,
                 const std::vector<size_t> &shot_range,
                 [[maybe_unused]] size_t bin_size = 0,
                 [[maybe_unused]] bool counts = false) {
        auto obs_name = obs.getObsName();
        auto obs_wires = obs.getWires();
        const size_t num_qubits = _statevector.getTotalNumQubits();

        StateVectorT sv(_statevector);

        if (obs_name.find("PauliX") != std::string::npos) {
            sv.applyOperation("Hadamard", obs_wires, false);
        } else if (obs_name.find("PauliY") != std::string::npos) {
            sv.applyOperation("PauliZ", obs_wires, false);
            sv.applyOperation("S", obs_wires, false);
            sv.applyOperation("Hadamard", obs_wires, false);
        } else if (obs_name.find("Hadamard") != std::string::npos) {
            const PrecisionT theta = -M_PI / 4.0;
            sv.applyOperation("RY", obs_wires, false, {theta});
        } else if (obs_name.find("PauliZ")!= std::string::npos) {
        }

        Derived measure(sv);

        std::vector<size_t> samples = measure.generate_samples(num_shots);
        std::vector<size_t> sub_samples;
        std::vector<PrecisionT> obs_samples(num_shots * obs_wires.size(), 0);

        if (shot_range.empty()) {
            sub_samples = samples;
        } else {
            // Get a slice of samples based on the shot_range vector
            for (auto &i : shot_range) {
                for (size_t j = i * num_qubits; j < (i + 1) * num_qubits; j++) {
                    sub_samples.push_back(samples[j]);
                }
            }
        }

        for (size_t i = 0; i < num_shots; i++) {
            obs_samples[i] =
                (1 - 2 * static_cast<PrecisionT>(
                             sub_samples[i * num_qubits + obs_wires[0]]));
        }
        return obs_samples;
    }
};

} // namespace Pennylane::Measures