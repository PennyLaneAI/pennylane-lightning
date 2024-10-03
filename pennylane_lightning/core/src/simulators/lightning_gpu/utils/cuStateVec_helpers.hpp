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

// Adapted from JET: https://github.com/XanaduAI/jet.git

/**
 * @file cudaStateVec_helpers.hpp
 */

#pragma once
#include <custatevec.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuStateVecError.hpp"

namespace Pennylane::LightningGPU::Util {

inline static auto pauliStringToEnum(const std::string &pauli_word)
    -> std::vector<custatevecPauli_t> {
    // Map string rep to Pauli enums
    const std::unordered_map<std::string, custatevecPauli_t> pauli_map{
        std::pair<const std::string, custatevecPauli_t>{std::string("I"),
                                                        CUSTATEVEC_PAULI_I},
        std::pair<const std::string, custatevecPauli_t>{std::string("X"),
                                                        CUSTATEVEC_PAULI_X},
        std::pair<const std::string, custatevecPauli_t>{std::string("Y"),
                                                        CUSTATEVEC_PAULI_Y},
        std::pair<const std::string, custatevecPauli_t>{std::string("Z"),
                                                        CUSTATEVEC_PAULI_Z}};

    constexpr std::size_t num_char = 1;

    std::vector<custatevecPauli_t> output;
    output.reserve(pauli_word.size());

    for (const auto &ch : pauli_word) {
        auto out = pauli_map.at(std::string(num_char, ch));
        output.push_back(out);
    }
    return output;
}

inline static auto pauliStringToOpNames(const std::string &pauli_word)
    -> std::vector<std::string> {
    // Map string rep to Pauli
    const std::unordered_map<std::string, std::string> pauli_map{
        std::pair<const std::string, std::string>{std::string("I"),
                                                  std::string("Identity")},
        std::pair<const std::string, std::string>{std::string("X"),
                                                  std::string("PauliX")},
        std::pair<const std::string, std::string>{std::string("Y"),
                                                  std::string("PauliY")},
        std::pair<const std::string, std::string>{std::string("Z"),
                                                  std::string("PauliZ")}};

    static constexpr std::size_t num_char = 1;

    std::vector<std::string> output;
    output.reserve(pauli_word.size());

    for (const auto ch : pauli_word) {
        auto out = pauli_map.at(std::string(num_char, ch));
        output.push_back(out);
    }
    return output;
}

/**
 * Utility function object to tell std::shared_ptr how to
 * release/destroy cuStateVec objects.
 */
struct handleDeleter {
    void operator()(custatevecHandle_t handle) const {
        PL_CUSTATEVEC_IS_SUCCESS(custatevecDestroy(handle));
    }
};

using SharedCusvHandle =
    std::shared_ptr<std::remove_pointer<custatevecHandle_t>::type>;

/**
 * @brief Creates a SharedCusvHandle (a shared pointer to a custatevecHandle)
 */
inline SharedCusvHandle make_shared_cusv_handle() {
    custatevecHandle_t h;
    PL_CUSTATEVEC_IS_SUCCESS(custatevecCreate(&h));
    return {h, handleDeleter()};
}

/**
 * @brief Compute the local index from a given index in multi-gpu workflow
 *
 * @param index Global index of the target element.
 * @param num_qubits Number of wires within the local devices.
 *
 *  @return local_index Local index of the target element.
 */
inline std::size_t compute_local_index(const std::size_t index,
                                       const std::size_t num_qubits) {
    // TODO: bound check for the left shift operation here
    constexpr std::size_t one{1U};
    const std::size_t local_index =
        (index >> num_qubits) * (one << num_qubits) ^ index;
    return local_index;
}

} // namespace Pennylane::LightningGPU::Util
