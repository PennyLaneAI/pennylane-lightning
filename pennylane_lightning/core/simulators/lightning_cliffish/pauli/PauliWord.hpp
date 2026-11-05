// Copyright 2026 Xanadu Quantum Technologies Inc.

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
 * @file PauliWord.hpp
 * A host-side Pauli word with phase, built on PauliKernels.hpp.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "Error.hpp"
#include "PauliKernels.hpp"

namespace Pennylane::LightningCliffish::Pauli {

/**
 * @brief A Pauli word i^log_i P on a fixed number of qubits.
 *
 * Character q of a string form is the Pauli on qubit q.
 */
class PauliWord {
  private:
    std::size_t num_qubits_;
    std::vector<Word> x_;
    std::vector<Word> z_;
    std::uint8_t log_i_{0};

  public:
    /**
     * @brief The identity on `num_qubits` qubits.
     */
    explicit PauliWord(std::size_t num_qubits)
        : num_qubits_{num_qubits}, x_(Pauli::numWords(num_qubits), 0),
          z_(Pauli::numWords(num_qubits), 0) {}

    /**
     * @brief Parse a word of the characters I, X, Y and Z.
     *
     * An optional leading "+", "-", "i" or "-i" sets the phase.
     */
    static auto fromString(std::string_view word) -> PauliWord {
        std::uint8_t log_i = 0;
        if (word.starts_with("-i")) {
            log_i = 3;
            word.remove_prefix(2);
        } else if (word.starts_with("+i") || word.starts_with("i")) {
            log_i = 1;
            word.remove_prefix(word.front() == '+' ? 2 : 1);
        } else if (word.starts_with("-")) {
            log_i = 2;
            word.remove_prefix(1);
        } else if (word.starts_with("+")) {
            word.remove_prefix(1);
        }

        PauliWord pauli(word.size());
        for (std::size_t qubit = 0; qubit < word.size(); qubit++) {
            pauli.set(qubit, word[qubit]);
        }
        pauli.log_i_ = log_i;
        return pauli;
    }

    [[nodiscard]] auto numQubits() const -> std::size_t { return num_qubits_; }

    [[nodiscard]] auto numWords() const -> std::size_t { return x_.size(); }

    [[nodiscard]] auto x() const -> const Word * { return x_.data(); }
    [[nodiscard]] auto z() const -> const Word * { return z_.data(); }
    [[nodiscard]] auto x() -> Word * { return x_.data(); }
    [[nodiscard]] auto z() -> Word * { return z_.data(); }

    /// Phase exponent: the word is i^logI() times its bare Pauli.
    [[nodiscard]] auto logI() const -> std::uint8_t { return log_i_; }
    void setLogI(std::uint8_t log_i) { log_i_ = log_i % 4; }

    /// A Pauli word is Hermitian when its phase is +1 or -1.
    [[nodiscard]] auto isHermitian() const -> bool { return log_i_ % 2 == 0; }

    /**
     * @brief The Pauli on `qubit`: one of 'I', 'X', 'Y' and 'Z'.
     */
    [[nodiscard]] auto get(std::size_t qubit) const -> char {
        PL_ABORT_IF_NOT(qubit < num_qubits_, "Qubit index out of range");
        const bool x_bit = getBit(x_.data(), qubit);
        const bool z_bit = getBit(z_.data(), qubit);
        if (x_bit) {
            return z_bit ? 'Y' : 'X';
        }
        return z_bit ? 'Z' : 'I';
    }

    /**
     * @brief Set the Pauli on `qubit` to one of 'I', 'X', 'Y' and 'Z'.
     */
    void set(std::size_t qubit, char pauli) {
        PL_ABORT_IF_NOT(qubit < num_qubits_, "Qubit index out of range");
        PL_ABORT_IF_NOT(pauli == 'I' || pauli == 'X' || pauli == 'Y' ||
                            pauli == 'Z',
                        "A Pauli word may only contain I, X, Y and Z");
        setBit(x_.data(), qubit, pauli == 'X' || pauli == 'Y');
        setBit(z_.data(), qubit, pauli == 'Z' || pauli == 'Y');
    }

    /// Number of qubits on which the word is not the identity.
    [[nodiscard]] auto weight() const -> std::size_t {
        std::size_t count = 0;
        for (std::size_t w = 0; w < x_.size(); w++) {
            count += static_cast<std::size_t>(Kokkos::popcount(x_[w] | z_[w]));
        }
        return count;
    }

    [[nodiscard]] auto commutes(const PauliWord &other) const -> bool {
        PL_ABORT_IF_NOT(num_qubits_ == other.num_qubits_,
                        "Pauli words act on different numbers of qubits");
        return !anticommutes(x_.data(), z_.data(), other.x_.data(),
                             other.z_.data(), x_.size());
    }

    /**
     * @brief Overwrite this word with the product (*this) * other.
     */
    auto operator*=(const PauliWord &other) -> PauliWord & {
        PL_ABORT_IF_NOT(num_qubits_ == other.num_qubits_,
                        "Pauli words act on different numbers of qubits");
        const int count = multiplyInPlace(x_.data(), z_.data(), other.x_.data(),
                                          other.z_.data(), x_.size());
        log_i_ = logIMod4(log_i_ + other.log_i_ + count);
        return *this;
    }

    friend auto operator*(PauliWord lhs, const PauliWord &rhs) -> PauliWord {
        lhs *= rhs;
        return lhs;
    }

    friend auto operator==(const PauliWord &lhs, const PauliWord &rhs)
        -> bool = default;

    /**
     * @brief The word as a string with its phase, e.g. "+XYZ" or "-iXX".
     */
    [[nodiscard]] auto toString() const -> std::string {
        static constexpr const char *phases[] = {"+", "+i", "-", "-i"};
        std::string word{phases[log_i_]};
        for (std::size_t qubit = 0; qubit < num_qubits_; qubit++) {
            word += get(qubit);
        }
        return word;
    }
};

} // namespace Pennylane::LightningCliffish::Pauli
