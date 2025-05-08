// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "UtilKokkos.hpp"

namespace Pennylane::LightningKokkos::Functors {

/**
 * @brief Compute probability distribution from StateVector.
 *
 * @tparam PrecisionT StateVector precision.
 * @tparam DeviceType Kokkos execution space.
 * @param arr_ StateVector data.
 * @param wires_ Wires for which the probability is computed.
 * @param all_indices_ Base indices.
 * @param all_offsets_ Offset indices.
 */
template <class PrecisionT, class DeviceType> class getProbsFunctor {
  public:
    // Required for functor:
    using execution_space = DeviceType;
    using value_type = PrecisionT[];
    const std::size_t value_count;

    using ComplexT = Kokkos::complex<PrecisionT>;
    Kokkos::View<ComplexT *> arr;
    Kokkos::View<std::size_t *> all_indices;
    Kokkos::View<std::size_t *> all_offsets;
    getProbsFunctor(const Kokkos::View<ComplexT *> &arr_,
                    const std::vector<std::size_t> &wires_,
                    const Kokkos::View<std::size_t *> all_indices_,
                    const Kokkos::View<std::size_t *> all_offsets_)
        : value_count{1U << wires_.size()}, arr{arr_},
          all_indices{all_indices_}, all_offsets{all_offsets_} {}

    KOKKOS_INLINE_FUNCTION
    void init(PrecisionT dst[]) const {
        for (std::size_t i = 0; i < value_count; i++)
            dst[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(PrecisionT dst[], const PrecisionT src[]) const {
        for (std::size_t i = 0; i < value_count; i++)
            dst[i] += src[i];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i, const std::size_t j,
                    PrecisionT dst[]) const {
        const std::size_t index = all_indices(i) + all_offsets(j);
        const PrecisionT rsv = arr(index).real();
        const PrecisionT isv = arr(index).imag();
        dst[i] += rsv * rsv + isv * isv;
    }
};

/**
 * @brief Compute probability distribution from StateVector.
 *
 * @tparam PrecisionT StateVector precision.
 * @tparam DeviceType Kokkos execution space.
 * @tparam num_wires Number of wires (0 is used for a dynamic number of wires).
 * @param arr_ StateVector data.
 * @param num_qubits_ Number of qubits.
 * @param wires_ Wires for which the probability is computed.
 */
template <class PrecisionT, class DeviceType, std::size_t num_wires>
class getProbsNQubitOpFunctor {
  public:
    // Required for functor:
    using execution_space = DeviceType;
    using value_type = PrecisionT[];
    const std::size_t value_count;

    using UnmanagedSizeTHostView =
        Kokkos::View<std::size_t *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ComplexT = Kokkos::complex<PrecisionT>;
    using KokkosComplexVector = Kokkos::View<ComplexT *>;
    Kokkos::View<ComplexT *> arr;
    const std::size_t n_wires;

    Kokkos::View<std::size_t *> parity;
    Kokkos::View<std::size_t *> rev_wires;

    std::size_t rev_wire_0;
    std::size_t rev_wire_1;
    std::size_t rev_wire_2;
    std::size_t rev_wire_3;
    std::size_t rev_wire_4;
    std::size_t rev_wire_5;
    std::size_t rev_wire_6;
    std::size_t rev_wire_7;
    std::size_t rev_wire_8;
    std::size_t parity_0;
    std::size_t parity_1;
    std::size_t parity_2;
    std::size_t parity_3;
    std::size_t parity_4;
    std::size_t parity_5;
    std::size_t parity_6;
    std::size_t parity_7;
    std::size_t parity_8;
    std::size_t parity_9;

    getProbsNQubitOpFunctor(const Kokkos::View<ComplexT *> &arr_,
                            const std::size_t num_qubits_,
                            const std::vector<std::size_t> &wires_)
        : value_count{1U << wires_.size()}, arr{arr_}, n_wires{wires_.size()} {
        PL_ABORT_IF(num_wires != 0 && num_wires != n_wires,
                    "num_wires must be equal to n_wires.");
        std::vector<std::size_t> rev_wires_(n_wires);
        for (std::size_t k = 0; k < n_wires; k++) {
            rev_wires_[n_wires - 1 - k] = (num_qubits_ - 1) - wires_[k];
        }
        std::vector<std::size_t> parity_ =
            Pennylane::Util::revWireParity(rev_wires_);
        if constexpr (num_wires == 0) {
            rev_wires =
                Pennylane::LightningKokkos::Util::vector2view(rev_wires_);
            parity = Pennylane::LightningKokkos::Util::vector2view(parity_);
        }
        if constexpr (num_wires > 0) {
            rev_wire_0 = rev_wires_[0];
            parity_0 = parity_[0];
            parity_1 = parity_[1];
        }
        if constexpr (num_wires > 1) {
            rev_wire_1 = rev_wires_[1];
            parity_2 = parity_[2];
        }
        if constexpr (num_wires > 2) {
            rev_wire_2 = rev_wires_[2];
            parity_3 = parity_[3];
        }
        if constexpr (num_wires > 3) {
            rev_wire_3 = rev_wires_[3];
            parity_4 = parity_[4];
        }
        if constexpr (num_wires > 4) {
            rev_wire_4 = rev_wires_[4];
            parity_5 = parity_[5];
        }
        if constexpr (num_wires > 5) {
            rev_wire_5 = rev_wires_[5];
            parity_6 = parity_[6];
        }
        if constexpr (num_wires > 6) {
            rev_wire_6 = rev_wires_[6];
            parity_7 = parity_[7];
        }
        if constexpr (num_wires > 7) {
            rev_wire_7 = rev_wires_[7];
            parity_8 = parity_[8];
        }
        if constexpr (num_wires > 8) {
            rev_wire_8 = rev_wires_[8];
            parity_9 = parity_[9];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void init(PrecisionT dst[]) const {
        for (std::size_t i = 0; i < value_count; i++)
            dst[i] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION
    void join(PrecisionT dst[], const PrecisionT src[]) const {
        for (std::size_t i = 0; i < value_count; i++)
            dst[i] += src[i];
    }

    KOKKOS_INLINE_FUNCTION
    void apply0(const std::size_t i0, const std::size_t rev_wire,
                const std::size_t offset, PrecisionT dst[]) const {
        std::size_t i1;
        i1 = i0 | (0U << rev_wire);
        PrecisionT rsv = real(arr(i1));
        PrecisionT isv = imag(arr(i1));
        dst[offset + 0] += rsv * rsv + isv * isv;
        i1 = i0 | (1U << rev_wire);
        rsv = real(arr(i1));
        isv = imag(arr(i1));
        dst[offset + 1] += rsv * rsv + isv * isv;
    }

    KOKKOS_INLINE_FUNCTION
    void apply1(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t offset,
                PrecisionT dst[]) const {
        apply0(i0, rev_wire_0, 0 + offset, dst);
        apply0(i0 | (1U << rev_wire_1), rev_wire_0, 2 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply2(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t offset, PrecisionT dst[]) const {
        apply1(i0, rev_wire_0, rev_wire_1, 0 + offset, dst);
        apply1(i0 | (1U << rev_wire_2), rev_wire_0, rev_wire_1, 4 + offset,
               dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply3(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t offset,
                PrecisionT dst[]) const {
        apply2(i0, rev_wire_0, rev_wire_1, rev_wire_2, 0 + offset, dst);
        apply2(i0 | (1U << rev_wire_3), rev_wire_0, rev_wire_1, rev_wire_2,
               8 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply4(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t rev_wire_4,
                const std::size_t offset, PrecisionT dst[]) const {
        apply3(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, 0 + offset,
               dst);
        apply3(i0 | (1U << rev_wire_4), rev_wire_0, rev_wire_1, rev_wire_2,
               rev_wire_3, 16 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply5(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t rev_wire_4,
                const std::size_t rev_wire_5, const std::size_t offset,
                PrecisionT dst[]) const {
        apply4(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, rev_wire_4,
               0 + offset, dst);
        apply4(i0 | (1U << rev_wire_5), rev_wire_0, rev_wire_1, rev_wire_2,
               rev_wire_3, rev_wire_4, 32 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply6(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t rev_wire_4,
                const std::size_t rev_wire_5, const std::size_t rev_wire_6,
                const std::size_t offset, PrecisionT dst[]) const {
        apply5(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, rev_wire_4,
               rev_wire_5, 0 + offset, dst);
        apply5(i0 | (1U << rev_wire_6), rev_wire_0, rev_wire_1, rev_wire_2,
               rev_wire_3, rev_wire_4, rev_wire_5, 64 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply7(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t rev_wire_4,
                const std::size_t rev_wire_5, const std::size_t rev_wire_6,
                const std::size_t rev_wire_7, const std::size_t offset,
                PrecisionT dst[]) const {
        apply6(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, rev_wire_4,
               rev_wire_5, rev_wire_6, 0 + offset, dst);
        apply6(i0 | (1U << rev_wire_7), rev_wire_0, rev_wire_1, rev_wire_2,
               rev_wire_3, rev_wire_4, rev_wire_5, rev_wire_6, 128 + offset,
               dst);
    }

    KOKKOS_INLINE_FUNCTION
    void apply8(const std::size_t i0, const std::size_t rev_wire_0,
                const std::size_t rev_wire_1, const std::size_t rev_wire_2,
                const std::size_t rev_wire_3, const std::size_t rev_wire_4,
                const std::size_t rev_wire_5, const std::size_t rev_wire_6,
                const std::size_t rev_wire_7, const std::size_t rev_wire_8,
                const std::size_t offset, PrecisionT dst[]) const {
        apply7(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, rev_wire_4,
               rev_wire_5, rev_wire_6, rev_wire_7, 0 + offset, dst);
        apply7(i0 | (1U << rev_wire_8), rev_wire_0, rev_wire_1, rev_wire_2,
               rev_wire_3, rev_wire_4, rev_wire_5, rev_wire_6, rev_wire_7,
               256 + offset, dst);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(std::size_t k, PrecisionT dst[]) const {
        if constexpr (num_wires == 0) {
            std::size_t i0 = (k & parity[0]);
            for (std::size_t i = 1; i < n_wires + 1; i++) {
                i0 |= ((k << i) & parity[i]);
            }
            for (std::size_t inner_idx = 0; inner_idx < value_count;
                 inner_idx++) {
                std::size_t idx = i0;
                for (std::size_t i = 0; i < n_wires; i++) {
                    idx |= ((inner_idx & (one << i)) >> i) << rev_wires[i];
                }
                const PrecisionT rsv = real(arr(idx));
                const PrecisionT isv = imag(arr(idx));
                dst[inner_idx] += rsv * rsv + isv * isv;
            }
        }
        if constexpr (num_wires == 1) {
            const std::size_t i0 = ((k << 1U) & parity_1) | (k & parity_0);
            apply0(i0, rev_wire_0, 0, dst);
        }
        if constexpr (num_wires == 2) {
            const std::size_t i0 = ((k << 2U) & parity_2) |
                                   ((k << 1U) & parity_1) | (k & parity_0);
            apply1(i0, rev_wire_0, rev_wire_1, 0, dst);
        }
        if constexpr (num_wires == 3) {
            const std::size_t i0 = ((k << 3U) & parity_3) |
                                   ((k << 2U) & parity_2) |
                                   ((k << 1U) & parity_1) | (k & parity_0);
            apply2(i0, rev_wire_0, rev_wire_1, rev_wire_2, 0, dst);
        }
        if constexpr (num_wires == 4) {
            const std::size_t i0 = ((k << 4U) & parity_4) |
                                   ((k << 3U) & parity_3) |
                                   ((k << 2U) & parity_2) |
                                   ((k << 1U) & parity_1) | (k & parity_0);
            apply3(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3, 0, dst);
        }
        if constexpr (num_wires == 5) {
            const std::size_t i0 =
                ((k << 4U) & parity_5) | ((k << 4U) & parity_4) |
                ((k << 3U) & parity_3) | ((k << 2U) & parity_2) |
                ((k << 1U) & parity_1) | (k & parity_0);
            apply4(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3,
                   rev_wire_4, 0, dst);
        }
        if constexpr (num_wires == 6) {
            const std::size_t i0 =
                ((k << 6U) & parity_6) | ((k << 5U) & parity_5) |
                ((k << 4U) & parity_4) | ((k << 3U) & parity_3) |
                ((k << 2U) & parity_2) | ((k << 1U) & parity_1) |
                (k & parity_0);
            apply5(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3,
                   rev_wire_4, rev_wire_5, 0, dst);
        }
        if constexpr (num_wires == 7) {
            const std::size_t i0 =
                ((k << 7U) & parity_7) | ((k << 6U) & parity_6) |
                ((k << 5U) & parity_5) | ((k << 4U) & parity_4) |
                ((k << 3U) & parity_3) | ((k << 2U) & parity_2) |
                ((k << 1U) & parity_1) | (k & parity_0);
            apply6(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3,
                   rev_wire_4, rev_wire_5, rev_wire_6, 0, dst);
        }
        if constexpr (num_wires == 8) {
            const std::size_t i0 =
                ((k << 8U) & parity_8) | ((k << 7U) & parity_7) |
                ((k << 6U) & parity_6) | ((k << 5U) & parity_5) |
                ((k << 4U) & parity_4) | ((k << 3U) & parity_3) |
                ((k << 2U) & parity_2) | ((k << 1U) & parity_1) |
                (k & parity_0);
            apply7(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3,
                   rev_wire_4, rev_wire_5, rev_wire_6, rev_wire_7, 0, dst);
        }
        if constexpr (num_wires == 9) {
            const std::size_t i0 =
                ((k << 9U) & parity_9) | ((k << 8U) & parity_8) |
                ((k << 7U) & parity_7) | ((k << 6U) & parity_6) |
                ((k << 5U) & parity_5) | ((k << 4U) & parity_4) |
                ((k << 3U) & parity_3) | ((k << 2U) & parity_2) |
                ((k << 1U) & parity_1) | (k & parity_0);
            apply8(i0, rev_wire_0, rev_wire_1, rev_wire_2, rev_wire_3,
                   rev_wire_4, rev_wire_5, rev_wire_6, rev_wire_7, rev_wire_8,
                   0, dst);
        }
    }
};

/**
 * @brief Compute probability distribution from StateVector.
 *
 * @tparam DeviceType Kokkos execution space.
 * @tparam PrecisionT StateVector precision.
 * @param arr StateVector data.
 * @param num_qubits Number of qubits.
 * @param wires Wires for which the probability is computed.
 */
template <class DeviceType, class PrecisionT>
auto probs_bitshift_generic(
    const Kokkos::View<Kokkos::complex<PrecisionT> *> arr,
    const std::size_t num_qubits, const std::vector<std::size_t> &wires)
    -> std::vector<PrecisionT> {
    const std::size_t n_wires = wires.size();
    const std::size_t n_probs = Pennylane::Util::exp2(n_wires);
    Kokkos::View<PrecisionT *> d_probabilities("d_probabilities", n_probs);
    switch (n_wires) {
    case 1UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 1>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 2UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 2>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 3UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 3>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 4UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 4>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 5UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 5>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 6UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 6>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 7UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 7>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    case 8UL:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 8>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    default:
        Kokkos::parallel_reduce(
            exp2(num_qubits - n_wires),
            getProbsNQubitOpFunctor<PrecisionT, DeviceType, 0>(arr, num_qubits,
                                                               wires),
            d_probabilities);
        break;
    }
    return Pennylane::LightningKokkos::Util::view2vector(d_probabilities);
};

/**
 *@brief Sampling using Random_XorShift64_Pool
 *
 * @param samples_ Kokkos::View of the generated samples.
 * @param cdf_  Kokkos::View of cumulative probability distribution.
 * @param rand_pool_ The generatorPool.
 * @param num_qubits_ Number of qubits.
 * @param length_ Length of cumulative probability distribution.
 */

template <class PrecisionT, template <class ExecutionSpace> class GeneratorPool,
          class ExecutionSpace = Kokkos::DefaultExecutionSpace>
struct Sampler {
    Kokkos::View<std::size_t *> samples;
    Kokkos::View<PrecisionT *> cdf;
    GeneratorPool<ExecutionSpace> rand_pool;

    const std::size_t num_qubits;
    const std::size_t length;

    Sampler(Kokkos::View<std::size_t *> samples_,
            Kokkos::View<PrecisionT *> cdf_,
            GeneratorPool<ExecutionSpace> rand_pool_,
            const std::size_t num_qubits_, const std::size_t length_)
        : samples(samples_), cdf(cdf_), rand_pool(rand_pool_),
          num_qubits(num_qubits_), length(length_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t k) const {
        // Get a random number state from the pool for the active thread
        auto rand_gen = rand_pool.get_state();
        PrecisionT U_rand = rand_gen.drand(0.0, 1.0);
        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
        std::size_t index;

        // Binary search for the bin index of cumulative probability
        // distribution that generated random number U falls into.
        if (U_rand <= cdf(1)) {
            index = 0;
        } else {
            std::size_t low_idx = 1, high_idx = length;
            std::size_t mid_idx;
            PrecisionT cdf_t;
            while (high_idx - low_idx > 1) {
                mid_idx = high_idx - ((high_idx - low_idx) >> 1U);
                if (mid_idx == length)
                    cdf_t = 1;
                else
                    cdf_t = cdf(mid_idx);
                if (cdf_t < U_rand)
                    low_idx = mid_idx;
                else
                    high_idx = mid_idx;
            }
            index = high_idx - 1;
        }
        for (std::size_t j = 0; j < num_qubits; j++) {
            samples(k * num_qubits + (num_qubits - 1 - j)) = (index >> j) & 1U;
        }
    }
};

/**
 * @brief Determines the transposed index of a tensor stored linearly.
 *  This function assumes each axis will have a length of 2 (|0>, |1>).
 *
 * @param sorted_ind_wires Data of indices for transposition.
 * @param trans_index Data of indices after transposition.
 * @param max_index_sorted_ind_wires_ Length of sorted_ind_wires.
 */
struct getTransposedIndexFunctor {
    Kokkos::View<std::size_t *> sorted_ind_wires;
    Kokkos::View<std::size_t *> trans_index;
    const std::size_t max_index_sorted_ind_wires;
    getTransposedIndexFunctor(Kokkos::View<std::size_t *> sorted_ind_wires_,
                              Kokkos::View<std::size_t *> trans_index_,
                              const int length_sorted_ind_wires_)
        : sorted_ind_wires(sorted_ind_wires_), trans_index(trans_index_),
          max_index_sorted_ind_wires(length_sorted_ind_wires_ - 1) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i, const std::size_t j) const {
        const std::size_t axis = sorted_ind_wires(j);
        const std::size_t index = i / (1L << (max_index_sorted_ind_wires - j));
        const std::size_t sub_index = (index % 2)
                                      << (max_index_sorted_ind_wires - axis);
        Kokkos::atomic_add(&trans_index(i), sub_index);
    }
};

/**
 * @brief Template for the transposition of state tensors,
 * axes are assumed to have a length of 2 (|0>, |1>).
 *
 * @tparam T Tensor data type.
 * @param tensor Tensor to be transposed.
 * @param new_axes new axes distribution.
 * @return Transposed Tensor.
 */
template <class PrecisionT> struct getTransposedFunctor {
    Kokkos::View<PrecisionT *> transProb;
    Kokkos::View<PrecisionT *> probability;
    Kokkos::View<std::size_t *> trans_index;
    getTransposedFunctor(Kokkos::View<PrecisionT *> transProb_,
                         Kokkos::View<PrecisionT *> probability_,
                         Kokkos::View<std::size_t *> trans_index_)
        : transProb(transProb_), probability(probability_),
          trans_index(trans_index_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t i) const {
        const std::size_t new_index = trans_index(i);
        transProb(i) = probability(new_index);
    }
};

} // namespace Pennylane::LightningKokkos::Functors
