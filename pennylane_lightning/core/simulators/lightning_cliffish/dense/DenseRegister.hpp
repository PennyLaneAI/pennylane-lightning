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
 * @file DenseRegister.hpp
 * The dense register of the hybrid method: a state vector on the few virtual
 * qubits that carry the program's magic.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "Error.hpp"
#include "PauliKernels.hpp"
#include "PauliRows.hpp"

namespace Pennylane::LightningCliffish::Dense {

/**
 * @brief A Pauli word i^log_i X^x Z^z on the register, with bit p of x and z
 * on register qubit p. As in PauliKernels.hpp, x and z both set is Y.
 */
struct DensePauli {
    std::uint64_t x{0};
    std::uint64_t z{0};
    std::uint8_t log_i{0};
};

/// The largest register the 64-bit masks of DensePauli can address.
inline constexpr std::size_t max_register_qubits = 40;

/**
 * @brief Amplitudes of 2^a basis states; register qubit p is bit p of the
 * index.
 *
 * @tparam ExecSpace Kokkos execution space that holds the amplitudes.
 */
template <class ExecSpace = Kokkos::DefaultExecutionSpace> class DenseRegister {
  public:
    using Complex = Kokkos::complex<double>;
    using MemorySpace = typename ExecSpace::memory_space;
    using AmplitudesView = Kokkos::View<Complex *, MemorySpace>;

  private:
    using Policy = Kokkos::RangePolicy<ExecSpace>;

    std::size_t num_qubits_;
    AmplitudesView amplitudes_;

    /// Index `k` with a zero bit inserted at position `bit`.
    KOKKOS_INLINE_FUNCTION static auto insertZero(std::size_t k,
                                                  std::size_t bit)
        -> std::size_t {
        const std::size_t low = k & ((std::size_t{1} << bit) - 1);
        return ((k >> bit) << (bit + 1)) | low;
    }

    /// i^n for n in [0, 4).
    KOKKOS_INLINE_FUNCTION static auto powerOfI(int n) -> Complex {
        switch (n & 3) {
        case 0:
            return {1.0, 0.0};
        case 1:
            return {0.0, 1.0};
        case 2:
            return {-1.0, 0.0};
        default:
            return {0.0, -1.0};
        }
    }

    /**
     * @brief The phase c(k) in P|k> = c(k) |k xor x>.
     *
     * Bare Y is i X Z, so each Y adds a factor of i.
     */
    KOKKOS_INLINE_FUNCTION static auto pauliPhase(const DensePauli &pauli,
                                                  std::size_t k) -> Complex {
        const int num_y = Kokkos::popcount(pauli.x & pauli.z);
        const int num_minus =
            Kokkos::popcount(static_cast<std::uint64_t>(k) & pauli.z);
        return powerOfI(pauli.log_i + num_y + 2 * num_minus);
    }

    void checkPauli(const DensePauli &pauli) const {
        const std::uint64_t mask = (num_qubits_ == 64)
                                       ? ~std::uint64_t{0}
                                       : (std::uint64_t{1} << num_qubits_) - 1;
        PL_ABORT_IF((pauli.x | pauli.z) & ~mask,
                    "Pauli word acts outside the dense register");
    }

  public:
    /**
     * @brief The state |0...0> on `num_qubits` qubits.
     */
    explicit DenseRegister(std::size_t num_qubits = 0)
        : num_qubits_{num_qubits} {
        PL_ABORT_IF(num_qubits > max_register_qubits,
                    "The dense register is too large");
        Pauli::ensureKokkosInitialized();
        amplitudes_ =
            AmplitudesView("dense_register", std::size_t{1} << num_qubits);
        Kokkos::deep_copy(Kokkos::subview(amplitudes_, 0), Complex{1.0, 0.0});
    }

    [[nodiscard]] auto numQubits() const -> std::size_t { return num_qubits_; }

    [[nodiscard]] auto amplitudes() const -> AmplitudesView {
        return amplitudes_;
    }

    /// The amplitudes, copied to the host.
    [[nodiscard]] auto hostAmplitudes() const -> std::vector<Complex> {
        std::vector<Complex> result(amplitudes_.size());
        using HostView = Kokkos::View<Complex *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        Kokkos::deep_copy(HostView(result.data(), result.size()), amplitudes_);
        return result;
    }

    /**
     * @brief Overwrite the state psi with a psi + b P psi.
     */
    void applyLinear(const DensePauli &pauli, Complex a, Complex b) {
        checkPauli(pauli);
        auto amplitudes = amplitudes_;
        const DensePauli word = pauli;
        if (word.x == 0) {
            Kokkos::parallel_for(
                "DenseRegister::applyDiagonal", Policy(0, amplitudes.size()),
                KOKKOS_LAMBDA(std::size_t k) {
                    amplitudes(k) *= a + b * pauliPhase(word, k);
                });
            return;
        }
        // Pair each index k0 with k1 = k0 xor x, where k0 has a zero at the
        // highest bit of x.
        const auto pivot =
            static_cast<std::size_t>(63 - Kokkos::countl_zero(word.x));
        Kokkos::parallel_for(
            "DenseRegister::applyPairs", Policy(0, amplitudes.size() / 2),
            KOKKOS_LAMBDA(std::size_t t) {
                const std::size_t k0 = insertZero(t, pivot);
                const std::size_t k1 = k0 ^ static_cast<std::size_t>(word.x);
                const Complex v0 = amplitudes(k0);
                const Complex v1 = amplitudes(k1);
                amplitudes(k0) = a * v0 + b * pauliPhase(word, k1) * v1;
                amplitudes(k1) = a * v1 + b * pauliPhase(word, k0) * v0;
            });
    }

    /**
     * @brief Apply exp(-i half_angle P) for a Hermitian P.
     */
    void rotate(const DensePauli &pauli, double half_angle) {
        applyLinear(pauli, {std::cos(half_angle), 0.0},
                    {0.0, -std::sin(half_angle)});
    }

    /**
     * @brief Apply exp(-i quarter_turns pi/4 P) exactly, for a Hermitian P.
     */
    void rotateQuarterTurns(const DensePauli &pauli,
                            std::uint8_t quarter_turns) {
        static constexpr double half = 0.70710678118654752440;
        switch (quarter_turns % 4) {
        case 1:
            applyLinear(pauli, {half, 0.0}, {0.0, -half});
            break;
        case 2:
            applyLinear(pauli, {0.0, 0.0}, {0.0, -1.0});
            break;
        case 3:
            applyLinear(pauli, {-half, 0.0}, {0.0, -half});
            break;
        default:
            break;
        }
    }

    /**
     * @brief <psi|P|psi> for a Hermitian P.
     */
    [[nodiscard]] auto expectation(const DensePauli &pauli) const -> double {
        checkPauli(pauli);
        auto amplitudes = amplitudes_;
        const DensePauli word = pauli;
        double value = 0.0;
        // (P psi)[k] = c(k xor x) psi[k xor x].
        Kokkos::parallel_reduce(
            "DenseRegister::expectation", Policy(0, amplitudes.size()),
            KOKKOS_LAMBDA(std::size_t k, double &local) {
                const std::size_t source = k ^ static_cast<std::size_t>(word.x);
                local += (Kokkos::conj(amplitudes(k)) *
                          pauliPhase(word, source) * amplitudes(source))
                             .real();
            },
            value);
        return value;
    }

    /**
     * @brief ||psi - P psi|| / (2 ||psi||): zero exactly when P stabilizes
     * psi, and linear in a small rotation away from that.
     */
    [[nodiscard]] auto residual(const DensePauli &pauli) const -> double {
        checkPauli(pauli);
        auto amplitudes = amplitudes_;
        const DensePauli word = pauli;
        double distance = 0.0;
        double norm = 0.0;
        Kokkos::parallel_reduce(
            "DenseRegister::residual", Policy(0, amplitudes.size()),
            KOKKOS_LAMBDA(std::size_t k, double &local_distance,
                          double &local_norm) {
                const std::size_t source = k ^ static_cast<std::size_t>(word.x);
                const Complex difference =
                    amplitudes(k) -
                    pauliPhase(word, source) * amplitudes(source);
                local_distance +=
                    Kokkos::abs(difference) * Kokkos::abs(difference);
                local_norm +=
                    Kokkos::abs(amplitudes(k)) * Kokkos::abs(amplitudes(k));
            },
            distance, norm);
        return std::sqrt(distance / norm) / 2;
    }

    /**
     * @brief Append a qubit in |0>, as the new highest bit.
     */
    void addQubit() {
        PL_ABORT_IF(num_qubits_ + 1 > max_register_qubits,
                    "The dense register is too large");
        AmplitudesView next("dense_register", 2 * amplitudes_.size());
        Kokkos::deep_copy(
            Kokkos::subview(
                next, Kokkos::make_pair(std::size_t{0}, amplitudes_.size())),
            amplitudes_);
        amplitudes_ = next;
        num_qubits_++;
    }

    /**
     * @brief Remove qubit `qubit`, which must be in |0>; the qubits above it
     * move down by one.
     *
     * @return The norm squared of the discarded |1> part, before the state is
     * renormalised.
     */
    auto dropQubit(std::size_t qubit) -> double {
        PL_ABORT_IF_NOT(qubit < num_qubits_, "Qubit index out of range");
        auto old_amplitudes = amplitudes_;
        AmplitudesView next("dense_register", amplitudes_.size() / 2);
        const std::uint64_t bit = std::uint64_t{1} << qubit;
        double kept = 0.0;
        double leaked = 0.0;
        Kokkos::parallel_reduce(
            "DenseRegister::dropQubit", Policy(0, next.size()),
            KOKKOS_LAMBDA(std::size_t t, double &local_kept,
                          double &local_leaked) {
                const std::size_t k0 = insertZero(t, qubit);
                const Complex v0 = old_amplitudes(k0);
                const Complex v1 = old_amplitudes(k0 | bit);
                next(t) = v0;
                local_kept += Kokkos::abs(v0) * Kokkos::abs(v0);
                local_leaked += Kokkos::abs(v1) * Kokkos::abs(v1);
            },
            kept, leaked);
        const double scale = (kept > 0.0) ? 1.0 / std::sqrt(kept) : 0.0;
        Kokkos::parallel_for(
            "DenseRegister::renormalise", Policy(0, next.size()),
            KOKKOS_LAMBDA(std::size_t t) { next(t) *= scale; });
        amplitudes_ = next;
        num_qubits_--;
        return leaked / (kept + leaked);
    }

    /// An independent copy of the register.
    [[nodiscard]] auto clone() const -> DenseRegister {
        DenseRegister copy(0);
        copy.num_qubits_ = num_qubits_;
        copy.amplitudes_ = AmplitudesView("dense_register", amplitudes_.size());
        Kokkos::deep_copy(copy.amplitudes_, amplitudes_);
        return copy;
    }
};

} // namespace Pennylane::LightningCliffish::Dense
