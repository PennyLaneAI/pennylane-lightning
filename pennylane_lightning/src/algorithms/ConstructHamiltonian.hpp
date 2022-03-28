// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "AdjointDiff.hpp"
#include "Macros.hpp"

#include <map>

namespace Pennylane::Algorithms {

class PauliWord {
  private:
    std::vector<std::pair<size_t, char>>
        pstring_; // vector of wire ('X', 'Y', 'Z') pair. All wires must be
                  // distinct

  public:
    PauliWord(std::string_view pstr, const std::vector<size_t> &wires) {
        if (pstr.size() != wires.size()) {
            throw std::invalid_argument(
                "Size of the Pauli string must be same as that of wires.");
        }
        for (size_t idx = 0; idx < pstr.size(); idx++) {
            if (pstr[idx] != 'X' || pstr[idx] != 'Y' || pstr[idx] != 'Z') {
                throw std::invalid_argument(
                    "An element of Pauli string must be one of X, Y, Z.");
            }
            pstring_.emplace_back(wires[idx], pstr[idx]);
        }
    }

    template <typename T>
    auto applyBasisState(size_t n_qubits, size_t basis_st) const
        -> std::pair<std::complex<T>, size_t> {
        // basis_st in binary rep
        uint32_t powi = 0;
        for (const auto &[wire, pchar] : pstring_) {
            switch (pchar) {
            case 'X':
                basis_st ^= (size_t{1U} << (n_qubits - wire - 1));
                break;
            case 'Y':
                powi += ((basis_st >> (n_qubits - wire - 1)) & 1U) == 0 ? 1 : 3;
                basis_st ^= (size_t{1U} << (n_qubits - wire - 1));
                break;
            case 'Z':
                powi += ((basis_st >> (n_qubits - wire - 1)) & 1U) == 0 ? 0 : 2;
                break;
            default:
                PL_UNREACHABLE;
            }
        }
        std::complex<T> scalar;
        switch (powi % 4) {
        case 0:
            scalar = std::complex<T>{1.0, 0.0};
            break;
        case 1:
            scalar = std::complex<T>{0.0, 1.0};
            break;
        case 2:
            scalar = std::complex<T>{-1.0, 0.0};
            break;
        case 3:
            scalar = std::complex<T>{0.0, -1.0};
            break;
        }
        return std::make_pair(scalar, basis_st);
    }

    [[nodiscard]] auto isDiagonal() const -> bool {
        return std::all_of(pstring_.begin(), pstring_.end(),
                           [](const auto &p) { return p.second == 'Z'; });
    }
};

template <typename T> class EfficientHamlitonian {
  private:
    size_t num_qubits_;
    std::vector<T> coeffs_;
    std::vector<PauliWord> pwords_;

  public:
    Hamlitonian(size_t num_qubits, std::vector<T> coeff,
                std::vector<PauliWord> pwords)
        : num_qubits_{num_qubits}, coeffs_{std::move(coeff)}, pwords_{std::move(
                                                                  pwords)} {}

    /**
     * @brief For a given binary representation of a basis state,
     * this function returns a pair of coefficients and basis states.
     */
    auto applyBasisState(size_t state_in) const
        -> std::pair<std::vector<std::complex<T>>, std::vector<size_t>> {
        std::vector<std::complex<T>> state_coeffs;
        std::vector<size_t> state_outs;
        state_coeffs.resize(pwords_.size());
        state_outs.resize(pwords_.size());
        for (size_t idx = 0; idx < pwords_.size(); idx++) {
            const auto [scalar, state_out] =
                pwords_[idx].applyBasisState<T>(num_qubits_, state_in);
            state_coeffs.emplace_back(scalar * coeffs_[idx]);
            state_outs.emplace_back(state_out);
        }
        return std::make_pair(state_coeffs, state_outs);
    }

    auto operator()(size_t state_in) const
        -> std::pair<std::vector<std::complex<T>>, std::vector<size_t>> {
        return this->applyBasisState(state_in);
    }

    [[nodiscard]] auto countTerms() const -> size_t { return pwords_.size(); }
};

/**
 * @brief Sparse matrix in csr format
 */
template <typename T> struct SparseMatrix {
    std::vector<T> data;
    std::vector<size_t> indices;
    std::vector<size_t> indptr;
};

template <typename T>
auto constructSparseMatrix(const EfficientHamlitonian<T> &ham)
    -> SparseMatrix<T> {
    SparseMatrix<T> result;

    result.indptr.emplace_back(0);

    for (size_t row = 0; row < (1U << ham.getNumQubits()); row++) {
        const auto [coeffs, cols] = ham(row);

        for (size_t idx = 0; idx < cols.size(); idx++) {
            result.data.emplace_back(coeffs[idx]);
        }
        result.indices.reserve(result.indies.size() + cols.size());
        result.indices.insert(result.indies.end(), cols.begin(), cols.end());
        result.indptr.emplace_back(result.indptr.back() + cols.size());
    }
    return result;
}
} // namespace Pennylane::Algorithms
