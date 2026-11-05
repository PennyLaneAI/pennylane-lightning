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
 * @file NativeTableau.hpp
 * The native tableau engine: an Aaronson-Gottesman tableau with
 * destabilizers, on Kokkos.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "AffineSubspace.hpp"
#include "Error.hpp"
#include "PauliKernels.hpp"
#include "PauliRows.hpp"
#include "PauliWord.hpp"
#include "TableauEngine.hpp"

namespace Pennylane::LightningCliffish::Tableau {

/**
 * @brief Aaronson-Gottesman tableau with destabilizers.
 *
 * For a state C|0...0>, row i is the destabilizer C X_i C^† and row n + i is
 * the stabilizer C Z_i C^†, each with its sign. Rows are generator-major and
 * bit-packed along qubits.
 *
 * @tparam ExecSpace Kokkos execution space that holds the tableau.
 */
template <class ExecSpace = Kokkos::DefaultExecutionSpace>
class NativeTableau final : public TableauEngine {
  private:
    using Word = Pauli::Word;
    using Rows = Pauli::PauliRows<ExecSpace>;
    using MemorySpace = typename Rows::MemorySpace;
    using Policy = Kokkos::RangePolicy<ExecSpace>;

    std::size_t num_qubits_;
    Rows rows_;
    /// One row holding the Pauli word of the op being applied.
    mutable Rows op_;

    /// Set rows [first, num_qubits_) to X_i and their stabilizers to Z_i.
    void initializeRows(std::size_t first) {
        auto x = rows_.x();
        auto z = rows_.z();
        const std::size_t num_qubits = num_qubits_;
        Kokkos::parallel_for(
            "NativeTableau::initializeRows", Policy(first, num_qubits),
            KOKKOS_LAMBDA(std::size_t i) {
                const Word bit = Word{1} << (i % Pauli::word_bits);
                x(i, i / Pauli::word_bits) = bit;
                z(num_qubits + i, i / Pauli::word_bits) = bit;
            });
    }

    void loadOp(const PauliWord &pauli) const {
        PL_ABORT_IF_NOT(pauli.numQubits() == num_qubits_,
                        "Pauli word acts on a different number of qubits "
                        "than the tableau");
        PL_ABORT_IF_NOT(pauli.isHermitian(), "Pauli word must be Hermitian");
        op_.setRow(0, pauli);
    }

    /**
     * @brief Flags, on the device, of rows [first, first + count) that
     * anticommute with the loaded op.
     */
    [[nodiscard]] auto anticommutingWithOp(std::size_t first,
                                           std::size_t count) const
        -> Kokkos::View<bool *, MemorySpace> {
        const Word *x = rows_.x().data();
        const Word *z = rows_.z().data();
        const Word *op_x = op_.x().data();
        const Word *op_z = op_.z().data();
        const std::size_t stride = rows_.x().stride(0);
        const std::size_t num_words = rows_.numWords();

        Kokkos::View<bool *, MemorySpace> flags("tableau_flags", count);
        Kokkos::parallel_for(
            "NativeTableau::anticommutingWithOp", Policy(0, count),
            KOKKOS_LAMBDA(std::size_t i) {
                const std::size_t row = first + i;
                flags(i) = Pauli::anticommutes(
                    x + row * stride, z + row * stride, op_x, op_z, num_words);
            });
        return flags;
    }

    /**
     * @brief Phase exponent of the product, in row order, of the stabilizers
     * whose destabilizers anticommute with the loaded op.
     *
     * When the op commutes with every stabilizer, this product is plus or
     * minus the op, and the two phases give its sign.
     */
    [[nodiscard]] auto stabilizerProductLogI() const -> std::uint8_t {
        const auto flags = anticommutingWithOp(0, num_qubits_);
        auto x = rows_.x();
        auto z = rows_.z();
        auto log_i = rows_.logI();
        const std::size_t num_qubits = num_qubits_;

        int count = 0;
        Kokkos::parallel_reduce(
            "NativeTableau::stabilizerProductWords",
            Policy(0, rows_.numWords()),
            KOKKOS_LAMBDA(std::size_t w, int &local_count) {
                Word acc_x = 0;
                Word acc_z = 0;
                for (std::size_t i = 0; i < num_qubits; i++) {
                    if (flags(i)) {
                        local_count += Pauli::productPhaseWord(
                            acc_x, acc_z, x(num_qubits + i, w),
                            z(num_qubits + i, w));
                        acc_x ^= x(num_qubits + i, w);
                        acc_z ^= z(num_qubits + i, w);
                    }
                }
            },
            count);

        int phases = 0;
        Kokkos::parallel_reduce(
            "NativeTableau::stabilizerProductPhases", Policy(0, num_qubits),
            KOKKOS_LAMBDA(std::size_t i, int &local_phases) {
                if (flags(i)) {
                    local_phases += log_i(num_qubits + i);
                }
            },
            phases);

        return Pauli::logIMod4(count + phases);
    }

    void applyRotation(std::uint8_t quarter_turns) {
        PL_ABORT_IF_NOT(quarter_turns >= 1 && quarter_turns <= 3,
                        "A rotation must have 1, 2 or 3 quarter turns");

        Word *x = rows_.x().data();
        Word *z = rows_.z().data();
        auto log_i = rows_.logI();
        const Word *op_x = op_.x().data();
        const Word *op_z = op_.z().data();
        auto op_log_i = op_.logI();
        const std::size_t stride = rows_.x().stride(0);
        const std::size_t num_words = rows_.numWords();

        // A row R that anticommutes with P maps to -R for a half turn, and to
        // -/+ i P R for one or three quarter turns.
        const int turn_log_i = (quarter_turns == 1) ? 3 : 1;
        Kokkos::parallel_for(
            "NativeTableau::applyRotation", Policy(0, rows_.numRows()),
            KOKKOS_LAMBDA(std::size_t row) {
                Word *row_x = x + row * stride;
                Word *row_z = z + row * stride;
                if (!Pauli::anticommutes(row_x, row_z, op_x, op_z, num_words)) {
                    return;
                }
                if (quarter_turns == 2) {
                    log_i(row) = Pauli::logIMod4(log_i(row) + 2);
                    return;
                }
                int count = 0;
                for (std::size_t w = 0; w < num_words; w++) {
                    count += Pauli::productPhaseWord(op_x[w], op_z[w], row_x[w],
                                                     row_z[w]);
                    row_x[w] ^= op_x[w];
                    row_z[w] ^= op_z[w];
                }
                log_i(row) = Pauli::logIMod4(log_i(row) + op_log_i(0) + count +
                                             turn_log_i);
            });
    }

    auto applyMeasurement(std::optional<bool> forced_outcome, std::mt19937 &gen)
        -> bool {
        const std::size_t num_qubits = num_qubits_;
        const auto stabilizer_flags =
            anticommutingWithOp(num_qubits, num_qubits);

        std::size_t pivot = std::numeric_limits<std::size_t>::max();
        Kokkos::parallel_reduce(
            "NativeTableau::measurementPivot", Policy(0, num_qubits),
            KOKKOS_LAMBDA(std::size_t i, std::size_t &local_pivot) {
                if (stabilizer_flags(i) && i < local_pivot) {
                    local_pivot = i;
                }
            },
            Kokkos::Min<std::size_t>(pivot));

        if (pivot >= num_qubits) {
            // Deterministic: the op is plus or minus a product of stabilizers.
            const std::uint8_t product_log_i = stabilizerProductLogI();
            std::uint8_t op_log_i = 0;
            Kokkos::deep_copy(op_log_i, Kokkos::subview(op_.logI(), 0));
            const bool outcome = product_log_i != op_log_i;
            PL_ABORT_IF(forced_outcome.has_value() &&
                            *forced_outcome != outcome,
                        "Cannot project onto a measurement outcome with zero "
                        "probability");
            return outcome;
        }

        const bool outcome = forced_outcome.has_value()
                                 ? *forced_outcome
                                 : ((gen() >> 31) & 1U) != 0;

        Word *x = rows_.x().data();
        Word *z = rows_.z().data();
        auto log_i = rows_.logI();
        const Word *op_x = op_.x().data();
        const Word *op_z = op_.z().data();
        auto op_log_i = op_.logI();
        const std::size_t stride = rows_.x().stride(0);
        const std::size_t num_words = rows_.numWords();
        const std::size_t pivot_row = num_qubits + pivot;

        // Multiply every other anticommuting row by the pivot stabilizer.
        Kokkos::parallel_for(
            "NativeTableau::measurementRows", Policy(0, rows_.numRows()),
            KOKKOS_LAMBDA(std::size_t row) {
                if (row == pivot_row) {
                    return;
                }
                Word *row_x = x + row * stride;
                Word *row_z = z + row * stride;
                if (!Pauli::anticommutes(row_x, row_z, op_x, op_z, num_words)) {
                    return;
                }
                const int count =
                    Pauli::multiplyInPlace(row_x, row_z, x + pivot_row * stride,
                                           z + pivot_row * stride, num_words);
                log_i(row) =
                    Pauli::logIMod4(log_i(row) + log_i(pivot_row) + count);
            });

        // The pivot stabilizer becomes its destabilizer, and the measured
        // word, with the outcome's sign, becomes the stabilizer.
        const int outcome_log_i = outcome ? 2 : 0;
        Kokkos::parallel_for(
            "NativeTableau::measurementPivotRows", Policy(0, num_words),
            KOKKOS_LAMBDA(std::size_t w) {
                x[pivot * stride + w] = x[pivot_row * stride + w];
                z[pivot * stride + w] = z[pivot_row * stride + w];
                x[pivot_row * stride + w] = op_x[w];
                z[pivot_row * stride + w] = op_z[w];
                if (w == 0) {
                    log_i(pivot) = log_i(pivot_row);
                    log_i(pivot_row) =
                        Pauli::logIMod4(op_log_i(0) + outcome_log_i);
                }
            });
        // A zero-word tableau has no word to carry the phase update.
        if (num_words == 0) {
            Kokkos::parallel_for(
                "NativeTableau::measurementPivotPhases", Policy(0, 1),
                KOKKOS_LAMBDA(std::size_t) {
                    log_i(pivot) = log_i(pivot_row);
                    log_i(pivot_row) =
                        Pauli::logIMod4(op_log_i(0) + outcome_log_i);
                });
        }
        return outcome;
    }

  public:
    /**
     * @brief The state |0...0> on `num_qubits` qubits.
     */
    explicit NativeTableau(std::size_t num_qubits = 0)
        : num_qubits_{num_qubits}, rows_(2 * num_qubits, num_qubits),
          op_(1, num_qubits) {
        initializeRows(0);
    }

    [[nodiscard]] auto name() const -> std::string override { return "native"; }

    [[nodiscard]] auto capabilities() const -> Capabilities override {
        return {};
    }

    [[nodiscard]] auto numQubits() const -> std::size_t override {
        return num_qubits_;
    }

    void addQubits(std::size_t count) override {
        if (count == 0) {
            return;
        }
        const std::size_t old_num_qubits = num_qubits_;
        const std::size_t new_num_qubits = num_qubits_ + count;
        Rows next(2 * new_num_qubits, new_num_qubits);

        auto old_x = rows_.x();
        auto old_z = rows_.z();
        auto old_log_i = rows_.logI();
        auto new_x = next.x();
        auto new_z = next.z();
        auto new_log_i = next.logI();
        const std::size_t old_num_words = rows_.numWords();
        Kokkos::parallel_for(
            "NativeTableau::addQubits", Policy(0, 2 * old_num_qubits),
            KOKKOS_LAMBDA(std::size_t row) {
                const std::size_t new_row =
                    (row < old_num_qubits) ? row : row + count;
                for (std::size_t w = 0; w < old_num_words; w++) {
                    new_x(new_row, w) = old_x(row, w);
                    new_z(new_row, w) = old_z(row, w);
                }
                new_log_i(new_row) = old_log_i(row);
            });

        rows_ = next;
        op_ = Rows(1, new_num_qubits);
        num_qubits_ = new_num_qubits;
        initializeRows(old_num_qubits);
    }

    auto applyBlock(const EngineBlock &block, std::mt19937 &gen)
        -> std::vector<bool> override {
        std::vector<bool> outcomes;
        for (const auto &op : block) {
            loadOp(op.pauli);
            if (op.kind == EngineOp::Kind::Rotation) {
                applyRotation(op.quarter_turns);
            } else {
                outcomes.push_back(applyMeasurement(op.forced_outcome, gen));
            }
        }
        return outcomes;
    }

    [[nodiscard]] auto expectation(const PauliWord &pauli) const
        -> int override {
        loadOp(pauli);
        const auto stabilizer_flags =
            anticommutingWithOp(num_qubits_, num_qubits_);
        int anticommuting = 0;
        Kokkos::parallel_reduce(
            "NativeTableau::expectation", Policy(0, num_qubits_),
            KOKKOS_LAMBDA(std::size_t i, int &local) {
                local += stabilizer_flags(i) ? 1 : 0;
            },
            anticommuting);
        if (anticommuting > 0) {
            return 0;
        }
        return stabilizerProductLogI() == pauli.logI() ? 1 : -1;
    }

    [[nodiscard]] auto pullBack(const PauliWord &pauli) const
        -> PauliWord override {
        loadOp(pauli);
        const std::size_t num_qubits = num_qubits_;
        // P = +/- prod_i D_i^a_i S_i^b_i, with a_i set when P anticommutes
        // with S_i and b_i set when it anticommutes with D_i.
        const auto destabilizer_flags = anticommutingWithOp(0, num_qubits);
        const auto stabilizer_flags =
            anticommutingWithOp(num_qubits, num_qubits);
        auto x = rows_.x();
        auto z = rows_.z();
        auto log_i = rows_.logI();

        int count = 0;
        Kokkos::parallel_reduce(
            "NativeTableau::pullBackWords", Policy(0, rows_.numWords()),
            KOKKOS_LAMBDA(std::size_t w, int &local_count) {
                Word acc_x = 0;
                Word acc_z = 0;
                for (std::size_t i = 0; i < num_qubits; i++) {
                    if (stabilizer_flags(i)) {
                        local_count += Pauli::productPhaseWord(
                            acc_x, acc_z, x(i, w), z(i, w));
                        acc_x ^= x(i, w);
                        acc_z ^= z(i, w);
                    }
                    if (destabilizer_flags(i)) {
                        local_count += Pauli::productPhaseWord(
                            acc_x, acc_z, x(num_qubits + i, w),
                            z(num_qubits + i, w));
                        acc_x ^= x(num_qubits + i, w);
                        acc_z ^= z(num_qubits + i, w);
                    }
                }
            },
            count);

        int phases = 0;
        Kokkos::parallel_reduce(
            "NativeTableau::pullBackPhases", Policy(0, num_qubits),
            KOKKOS_LAMBDA(std::size_t i, int &local_phases) {
                local_phases += stabilizer_flags(i) ? log_i(i) : 0;
                local_phases +=
                    destabilizer_flags(i) ? log_i(num_qubits + i) : 0;
            },
            phases);
        const std::uint8_t product_log_i = Pauli::logIMod4(count + phases);

        const auto host_a = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, stabilizer_flags);
        const auto host_b = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, destabilizer_flags);

        // C^† P C = i^(log P - log Q) prod_i X_i^a_i Z_i^b_i, and X Z = -i Y.
        PauliWord pulled(num_qubits);
        int xz_pairs = 0;
        for (std::size_t i = 0; i < num_qubits; i++) {
            Pauli::setBit(pulled.x(), i, host_a(i));
            Pauli::setBit(pulled.z(), i, host_b(i));
            xz_pairs += (host_a(i) && host_b(i)) ? 1 : 0;
        }
        pulled.setLogI(
            Pauli::logIMod4(pauli.logI() - product_log_i + 3 * xz_pairs));
        return pulled;
    }

    [[nodiscard]] auto
    computationalSupport(const std::vector<std::size_t> &qubits) const
        -> GF2::AffineSubspace override {
        for (auto qubit : qubits) {
            PL_ABORT_IF_NOT(qubit < num_qubits_, "Qubit index out of range");
        }
        const std::size_t num_qubits = num_qubits_;
        const auto host_x =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rows_.x());
        const auto host_z =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rows_.z());
        const auto host_log_i = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, rows_.logI());

        std::vector<PauliWord> stabilizers;
        stabilizers.reserve(num_qubits);
        for (std::size_t i = 0; i < num_qubits; i++) {
            PauliWord stabilizer(num_qubits);
            for (std::size_t w = 0; w < rows_.numWords(); w++) {
                stabilizer.x()[w] = host_x(num_qubits + i, w);
                stabilizer.z()[w] = host_z(num_qubits + i, w);
            }
            stabilizer.setLogI(host_log_i(num_qubits + i));
            stabilizers.push_back(std::move(stabilizer));
        }

        // Row-reduce the X parts. The rows left with no X part generate the
        // Z-type stabilizers.
        std::size_t rank = 0;
        for (std::size_t q = 0; q < num_qubits && rank < num_qubits; q++) {
            std::size_t pivot = rank;
            while (pivot < num_qubits &&
                   !Pauli::getBit(stabilizers[pivot].x(), q)) {
                pivot++;
            }
            if (pivot == num_qubits) {
                continue;
            }
            std::swap(stabilizers[rank], stabilizers[pivot]);
            for (std::size_t i = 0; i < num_qubits; i++) {
                if (i != rank && Pauli::getBit(stabilizers[i].x(), q)) {
                    stabilizers[i] *= stabilizers[rank];
                }
            }
            rank++;
        }

        // A stabilizer (-1)^s Z^a allows only the outcomes x with a.x = s.
        const std::size_t bit_words = GF2::numBitWords(num_qubits);
        std::vector<GF2::Bits> constraints;
        std::vector<bool> parities;
        for (std::size_t i = rank; i < num_qubits; i++) {
            const Word *z = stabilizers[i].z();
            constraints.emplace_back(z, z + bit_words);
            parities.push_back(stabilizers[i].logI() == 2);
        }
        const auto support = GF2::AffineSubspace::solve(
            num_qubits, std::move(constraints), std::move(parities));
        PL_ABORT_IF_NOT(support.has_value(),
                        "The tableau's Z-type stabilizers are inconsistent");
        return support->project(qubits);
    }

    [[nodiscard]] auto clone() const
        -> std::unique_ptr<TableauEngine> override {
        auto copy = std::make_unique<NativeTableau>(num_qubits_);
        Kokkos::deep_copy(copy->rows_.x(), rows_.x());
        Kokkos::deep_copy(copy->rows_.z(), rows_.z());
        Kokkos::deep_copy(copy->rows_.logI(), rows_.logI());
        return copy;
    }
};

} // namespace Pennylane::LightningCliffish::Tableau
