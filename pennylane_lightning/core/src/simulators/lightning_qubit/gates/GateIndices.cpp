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
#include "GateIndices.hpp"
#include "ConstantUtil.hpp"
#include "Util.hpp" // exp2, maxDecimalForQubit

namespace Pennylane::LightningQubit::Gates {

auto getIndicesAfterExclusion(const std::vector<std::size_t> &indicesToExclude,
                              std::size_t num_qubits)
    -> std::vector<std::size_t> {
    std::set<std::size_t> indices;
    for (size_t i = 0; i < num_qubits; i++) {
        indices.emplace(i);
    }
    for (const std::size_t &excludedIndex : indicesToExclude) {
        indices.erase(excludedIndex);
    }
    return {indices.begin(), indices.end()};
}

auto generateBitPatterns(const std::vector<std::size_t> &qubitIndices,
                         std::size_t num_qubits) -> std::vector<std::size_t> {
    std::vector<std::size_t> indices;
    indices.reserve(Pennylane::Util::exp2(qubitIndices.size()));
    indices.emplace_back(0);

    // NOLINTNEXTLINE(modernize-loop-convert)
    for (auto index_it = qubitIndices.rbegin(); index_it != qubitIndices.rend();
         index_it++) {
        const std::size_t value =
            Pennylane::Util::maxDecimalForQubit(*index_it, num_qubits);
        const std::size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}

/**
 * @brief Computes the CSR-format indices of a Pauli word times a coefficient.
 *
 * @param word Pauli word such as "XYYX".
 */
auto generatePauliWordIndices(const std::string &word)
    -> std::vector<std::size_t> {
    const std::size_t n_word = word.size();
    const std::size_t n_indices = Pennylane::Util::exp2(n_word);
    std::vector<std::size_t> indices(n_indices);
    std::vector<std::size_t> mtype(n_word);
    std::transform(word.begin(), word.end(), mtype.begin(), [](auto ch) {
        return (ch == 'I' || ch == 'Z') ? 0UL : 1UL;
    });
    std::size_t current_size = 2;
    indices[0] = 0UL;
    indices[1] = 1UL;
    if (mtype[n_word - 1] == 1) {
        std::swap(indices[0], indices[1]);
    }
    for (std::size_t i = 1; i < n_word; i++) {
        if (mtype[n_word - 1 - i] == 0) {
            for (std::size_t j = 0; j < current_size; j++) {
                indices[j + current_size] = indices[j] + current_size;
            }
        } else {
            std::copy(indices.begin(), indices.begin() + current_size,
                      indices.begin() + current_size);
            for (std::size_t j = 0; j < current_size; j++) {
                indices[j] += current_size;
            }
        }
        current_size *= 2;
    }
    return indices;
}

/**
 * @brief Computes the CSR-format data of a Pauli word times a coefficient.
 *
 * @tparam PrecisionT Floating point data type.
 * @param word Pauli word such as "XYYX".
 * @param coeff Multiplicative factor.
 */
template <class PrecisionT>
auto generatePauliWordData(const std::string &word,
                           const std::complex<PrecisionT> &coeff)
    -> std::vector<std::complex<PrecisionT>> {
    constexpr auto IMAG = Pennylane::Util::IMAG<PrecisionT>();
    const std::size_t n_word = word.size();
    const std::size_t n_data = Pennylane::Util::exp2(n_word);
    std::vector<std::complex<PrecisionT>> data(n_data);
    std::vector<std::size_t> mtype(n_word);
    std::transform(word.begin(), word.end(), mtype.begin(), [](auto ch) {
        return (ch == 'I' || ch == 'X') ? 0UL : ((ch == 'Y') ? 1UL : 2UL);
    });
    std::size_t current_size = 2;
    switch (mtype[n_word - 1 - 0]) {
    case 0:
        data[0] = 1.0;
        data[1] = 1.0;
        break;
    case 1:
        data[0] = -IMAG;
        data[1] = IMAG;
        break;
    default:
        data[0] = 1.0;
        data[1] = -1.0;
        break;
    }
    data[0] *= coeff;
    data[1] *= coeff;
    for (std::size_t i = 1; i < n_word; i++) {
        switch (mtype[n_word - 1 - i]) {
        case 0:
            std::copy(data.begin(), data.begin() + current_size,
                      data.begin() + current_size);
            break;
        case 1:
            for (std::size_t j = 0; j < current_size; j++) {
                data[j + current_size] = IMAG * data[j];
            }
            for (std::size_t j = 0; j < current_size; j++) {
                data[j] *= -IMAG;
            }
            break;
        default:
            for (std::size_t j = 0; j < current_size; j++) {
                data[j + current_size] = -data[j];
            }
            break;
        }
        current_size *= 2;
    }
    return data;
}
template std::vector<std::complex<float>>
generatePauliWordData<float>(const std::string &word,
                             const std::complex<float> &coeff);
template std::vector<std::complex<double>>
generatePauliWordData<double>(const std::string &word,
                              const std::complex<double> &coeff);

} // namespace Pennylane::LightningQubit::Gates
