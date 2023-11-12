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

#include <regex>
#include <string>
#include <vector>

#include "Observables.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Observables;
void parse_obs2ops(const std::string &obs_name, std::vector<std::string> &ops,
                   std::vector<std::vector<size_t>> &wires) {
    std::regex regex(R"((Pauli[XYZ]|Hadamard|Identity)\[(\d+)\])");
    // Use std::sregex_iterator to iterate over matches in the obs_name string
    auto it = std::sregex_iterator(obs_name.begin(), obs_name.end(), regex);
    auto end = std::sregex_iterator();

    for (; it != end; ++it) {
        std::smatch match = *it;
        ops.push_back(match[1].str());
        wires.push_back({std::stoul(match[2].str())});
    }
}

auto sample_to_str(std::vector<size_t> &sample) -> std::string {
    std::string str;
    for (auto &element : sample) {
        str += std::to_string(element);
    }
    return str;
}

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
    auto expval(const Observable<StateVectorT> &obs, const size_t &num_shots,
                const std::vector<size_t> &shot_range) -> PrecisionT {
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

        std::vector<std::string> ops;
        std::vector<std::vector<size_t>> wires_list;
        parse_obs2ops(obs_name, ops, wires_list);

        size_t num_identity_obs = 0;

        for (size_t i = 0; i < ops.size(); i++) {
            auto ops_name = ops[i];
            if (ops_name == "PauliX") {
                sv.applyOperation("Hadamard", wires_list[i], false);
            } else if (ops_name == "PauliY") {
                sv.applyOperations(
                    {"PauliZ", "S", "Hadamard"},
                    {wires_list[i], wires_list[i], wires_list[i]},
                    {false, false, false});
            } else if (ops_name == "Hadamard") {
                const PrecisionT theta = -M_PI / 4.0;
                sv.applyOperation("RY", wires_list[i], false, {theta});
            } else if (ops_name == "PauliZ") {
            } else if (ops_name == "Identity") {
                std::swap(obs_wires[num_identity_obs], obs_wires[i]);
                num_identity_obs++;
            }
        }

        Derived measure(sv);

        std::vector<size_t> samples = measure.generate_samples(num_shots);
        std::vector<size_t> sub_samples;
        std::vector<PrecisionT> obs_samples(num_shots, 0);

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
            std::vector<size_t> local_sample(obs_wires.size());
            for (size_t j = 0; j < obs_wires.size(); j++) {
                local_sample[j] = sub_samples[i * num_qubits + obs_wires[j]];
            }

            if (num_identity_obs != obs_wires.size()) {
                if (std::reduce(local_sample.begin() + num_identity_obs,
                                local_sample.end()) %
                        2 ==
                    1) {
                    obs_samples[i] = -1;
                } else {
                    obs_samples[i] = 1;
                }
            } else {
                obs_samples[i] = 1;
            }
        }
        return obs_samples;
    }

    /**
     * @brief Groups the samples into a dictionary showing number of occurences
     * for each possible outcome.
     *
     * @param samples A vector of samples with size of ``num_shots *
     * num_obs_wires``
     * @param num_wires number of wires the sampled observable was performed on
     *
     * @return std::unordered_map<std::string, size_t> with format ``{'outcome':
     * num_occurences}``
     */
    auto samples_to_counts(std::vector<size_t> &samples, size_t &num_shots,
                           size_t &num_obs_wires)
        -> std::unordered_map<std::string, size_t> {
        std::unordered_map<std::string, size_t> outcome_map;

        for (size_t i = 0; i < num_shots; i++) {
            auto local_sample =
                std::vector(samples.begin() + i * num_obs_wires,
                            samples.begin() + (i + 1) * num_obs_wires - 1);
            std::string key = sample_to_str(local_sample);

            auto it = outcome_map.find(key);

            if (it != outcome_map.end()) {
                it->second += 1;
            } else {
                outcome_map[key] = 1;
            }
        }
        return outcome_map;
    }
};

} // namespace Pennylane::Measures