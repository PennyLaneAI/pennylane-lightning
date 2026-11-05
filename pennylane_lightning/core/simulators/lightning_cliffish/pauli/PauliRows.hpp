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
 * @file PauliRows.hpp
 * A set of Pauli words stored row by row in Kokkos views.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

#include <Kokkos_Core.hpp>

#include "Error.hpp"
#include "PauliKernels.hpp"
#include "PauliWord.hpp"

namespace Pennylane::LightningCliffish::Pauli {

/**
 * @brief Initialize Kokkos once per process and finalize it at exit.
 */
inline void ensureKokkosInitialized() {
    static std::mutex init_mutex;
    static bool is_exit_registered = false;

    const std::lock_guard<std::mutex> lock(init_mutex);
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
        Kokkos::initialize();
    }
    if (!is_exit_registered) {
        is_exit_registered = true;
        std::atexit([]() {
            if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
                Kokkos::finalize();
            }
        });
    }
}

/**
 * @brief Pauli words on a common number of qubits, one word per row.
 *
 * Rows are generator-major: the x and z bits of a row are contiguous, so a
 * row operation streams through memory. The phase exponent of each row is
 * stored separately.
 *
 * @tparam ExecSpace Kokkos execution space that holds the rows.
 */
template <class ExecSpace = Kokkos::DefaultExecutionSpace> class PauliRows {
  public:
    using MemorySpace = typename ExecSpace::memory_space;
    using WordsView = Kokkos::View<Word **, Kokkos::LayoutRight, MemorySpace>;
    using PhasesView = Kokkos::View<std::uint8_t *, MemorySpace>;

  private:
    std::size_t num_rows_;
    std::size_t num_qubits_;
    std::size_t num_words_;
    WordsView x_;
    WordsView z_;
    PhasesView log_i_;

  public:
    /**
     * @brief `num_rows` identity words on `num_qubits` qubits.
     */
    PauliRows(std::size_t num_rows, std::size_t num_qubits)
        : num_rows_{num_rows}, num_qubits_{num_qubits},
          num_words_{Pauli::numWords(num_qubits)} {
        ensureKokkosInitialized();
        x_ = WordsView("pauli_rows_x", num_rows_, num_words_);
        z_ = WordsView("pauli_rows_z", num_rows_, num_words_);
        log_i_ = PhasesView("pauli_rows_log_i", num_rows_);
    }

    [[nodiscard]] auto numRows() const -> std::size_t { return num_rows_; }
    [[nodiscard]] auto numQubits() const -> std::size_t { return num_qubits_; }
    [[nodiscard]] auto numWords() const -> std::size_t { return num_words_; }

    [[nodiscard]] auto x() const -> WordsView { return x_; }
    [[nodiscard]] auto z() const -> WordsView { return z_; }
    [[nodiscard]] auto logI() const -> PhasesView { return log_i_; }

    /**
     * @brief Copy `word` into row `row`.
     */
    void setRow(std::size_t row, const PauliWord &word) {
        PL_ABORT_IF_NOT(row < num_rows_, "Row index out of range");
        PL_ABORT_IF_NOT(word.numQubits() == num_qubits_,
                        "Pauli word acts on a different number of qubits");

        using HostWords = Kokkos::View<const Word *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        Kokkos::deep_copy(Kokkos::subview(x_, row, Kokkos::ALL),
                          HostWords(word.x(), num_words_));
        Kokkos::deep_copy(Kokkos::subview(z_, row, Kokkos::ALL),
                          HostWords(word.z(), num_words_));
        Kokkos::deep_copy(Kokkos::subview(log_i_, row), word.logI());
    }

    /**
     * @brief Copy row `row` out as a host Pauli word.
     */
    [[nodiscard]] auto getRow(std::size_t row) const -> PauliWord {
        PL_ABORT_IF_NOT(row < num_rows_, "Row index out of range");

        using HostWords = Kokkos::View<Word *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        PauliWord word(num_qubits_);
        Kokkos::deep_copy(HostWords(word.x(), num_words_),
                          Kokkos::subview(x_, row, Kokkos::ALL));
        Kokkos::deep_copy(HostWords(word.z(), num_words_),
                          Kokkos::subview(z_, row, Kokkos::ALL));
        std::uint8_t log_i = 0;
        Kokkos::deep_copy(log_i, Kokkos::subview(log_i_, row));
        word.setLogI(log_i);
        return word;
    }

    /**
     * @brief Overwrite row `target` with the product row(target) * row(source).
     */
    void multiplyRow(std::size_t target, std::size_t source) {
        PL_ABORT_IF_NOT(target < num_rows_ && source < num_rows_,
                        "Row index out of range");
        PL_ABORT_IF(target == source, "Cannot multiply a row by itself");

        auto x = x_;
        auto z = z_;
        auto log_i = log_i_;
        int count = 0;
        Kokkos::parallel_reduce(
            "PauliRows::multiplyRow",
            Kokkos::RangePolicy<ExecSpace>(0, num_words_),
            KOKKOS_LAMBDA(std::size_t w, int &local_count) {
                local_count += productPhaseWord(x(target, w), z(target, w),
                                                x(source, w), z(source, w));
                x(target, w) ^= x(source, w);
                z(target, w) ^= z(source, w);
            },
            count);
        Kokkos::parallel_for(
            "PauliRows::multiplyRowPhase", Kokkos::RangePolicy<ExecSpace>(0, 1),
            KOKKOS_LAMBDA(std::size_t) {
                log_i(target) = logIMod4(log_i(target) + log_i(source) + count);
            });
    }

    /**
     * @brief Whether row `row` anticommutes with each other row, as a host
     * vector of flags indexed by row.
     */
    [[nodiscard]] auto anticommutingRows(std::size_t row) const
        -> Kokkos::View<bool *, Kokkos::HostSpace> {
        PL_ABORT_IF_NOT(row < num_rows_, "Row index out of range");

        const Word *x = x_.data();
        const Word *z = z_.data();
        const std::size_t stride = x_.stride(0);
        const std::size_t num_words = num_words_;
        Kokkos::View<bool *, MemorySpace> flags("pauli_rows_anticommuting",
                                                num_rows_);
        Kokkos::parallel_for(
            "PauliRows::anticommutingRows",
            Kokkos::RangePolicy<ExecSpace>(0, num_rows_),
            KOKKOS_LAMBDA(std::size_t other) {
                flags(other) = anticommutes(x + row * stride, z + row * stride,
                                            x + other * stride,
                                            z + other * stride, num_words);
            });
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, flags);
    }
};

} // namespace Pennylane::LightningCliffish::Pauli
