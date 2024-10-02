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
#include <chrono>
#include <cstdint>
#include <optional>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "ExpValFunctors.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
#include "MeasurementsBase.hpp"
#include "MeasuresFunctors.hpp"
#include "Observables.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Functors;
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using Pennylane::LightningKokkos::StateVectorKokkos;
using Pennylane::LightningKokkos::Util::getRealOfComplexInnerProduct;
using Pennylane::LightningKokkos::Util::SparseMV_Kokkos;
using Pennylane::LightningKokkos::Util::vector2view;
using Pennylane::LightningKokkos::Util::view2vector;
using Pennylane::Util::exp2;
enum class ExpValFunc : uint32_t {
    BEGIN = 1,
    Identity = 1,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    END
};
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Measures {
template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;
    using KokkosExecSpace = typename StateVectorT::KokkosExecSpace;
    using HostExecSpace = typename StateVectorT::HostExecSpace;
    using KokkosVector = typename StateVectorT::KokkosVector;
    using KokkosSizeTVector = typename StateVectorT::KokkosSizeTVector;
    using UnmanagedSizeTHostView =
        typename StateVectorT::UnmanagedSizeTHostView;
    using UnmanagedConstComplexHostView =
        typename StateVectorT::UnmanagedConstComplexHostView;
    using UnmanagedConstSizeTHostView =
        typename StateVectorT::UnmanagedConstSizeTHostView;
    using ScratchViewComplex = typename StateVectorT::ScratchViewComplex;
    using TeamPolicy = typename StateVectorT::TeamPolicy;

  public:
    explicit Measurements(const StateVectorT &statevector)
        : BaseType{statevector} {
        init_expval_funcs_();
    };

    /**
     * @brief Templated method that returns the expectation value of named
     * observables.
     *
     * @tparam functor_t Expectation value functor class for Kokkos dispatcher.
     * @tparam num_wires Number of wires.
     * @param wires Wires to apply the observable to.
     */
    template <template <class> class functor_t, int num_wires>
    auto applyExpValNamedFunctor(const std::vector<std::size_t> &wires)
        -> PrecisionT {
        if constexpr (num_wires > 0)
            PL_ASSERT(wires.size() == num_wires);

        const std::size_t num_qubits = this->_statevector.getNumQubits();
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0.0;
        Kokkos::parallel_reduce(exp2(num_qubits - num_wires),
                                functor_t(arr_data, num_qubits, wires), expval);
        return expval;
    }

    /**
     * @brief Templated method that returns the expectation value of a
     * matrix-valued operator.
     *
     * @tparam functor_t Expectation value functor class for Kokkos dispatcher.
     * @tparam num_wires Number of wires.
     * @param matrix Matrix (linearized into a KokkosVector).
     * @param wires Wires to apply the observable to.
     */
    template <template <class> class functor_t, int num_wires>
    auto applyExpValFunctor(const KokkosVector matrix,
                            const std::vector<std::size_t> &wires)
        -> PrecisionT {
        PL_ASSERT(wires.size() == num_wires);
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0.0;
        Kokkos::parallel_reduce(
            exp2(num_qubits - num_wires),
            functor_t<PrecisionT>(arr_data, num_qubits, matrix, wires), expval);
        return expval;
    }

    /**
     * @brief Calculate expectation value with respect to multi qubit observable
     * on specified wires.
     *
     * @param matrix Hermitian matrix representing observable to be used.
     * @param wires Wires to apply observable to.
     * @param params Not used.
     * @return Expectation value with respect to observable applied to specified
     * wires.
     */
    auto getExpValMatrix(const KokkosVector matrix,
                         const std::vector<std::size_t> &wires) -> PrecisionT {
        std::size_t num_qubits = this->_statevector.getNumQubits();
        std::size_t two2N = std::exp2(num_qubits - wires.size());
        std::size_t dim = std::exp2(wires.size());
        const KokkosVector arr_data = this->_statevector.getView();

        PrecisionT expval = 0.0;
        switch (wires.size()) {
        case 1:
            Kokkos::parallel_reduce(two2N,
                                    getExpVal1QubitOpFunctor<PrecisionT>(
                                        arr_data, num_qubits, matrix, wires),
                                    expval);
            break;
        case 2:
            Kokkos::parallel_reduce(two2N,
                                    getExpVal2QubitOpFunctor<PrecisionT>(
                                        arr_data, num_qubits, matrix, wires),
                                    expval);
            break;
        case 3:
            Kokkos::parallel_reduce(two2N,
                                    getExpVal3QubitOpFunctor<PrecisionT>(
                                        arr_data, num_qubits, matrix, wires),
                                    expval);
            break;
        case 4:
            Kokkos::parallel_reduce(two2N,
                                    getExpVal4QubitOpFunctor<PrecisionT>(
                                        arr_data, num_qubits, matrix, wires),
                                    expval);
            break;
        default:
            std::size_t scratch_size = ScratchViewComplex::shmem_size(dim);
            Kokkos::parallel_reduce(
                "getExpValMultiQubitOpFunctor",
                TeamPolicy(two2N, Kokkos::AUTO, dim)
                    .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
                getExpValMultiQubitOpFunctor<PrecisionT>(arr_data, num_qubits,
                                                         matrix, wires),
                expval);
            break;
        }
        return expval;
    }

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param obs An Observable object.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        StateVectorT ob_sv{this->_statevector};
        obs.applyInPlace(ob_sv);
        return getRealOfComplexInnerProduct(this->_statevector.getView(),
                                            ob_sv.getView());
    }

    /**
     * @brief Calculate expectation value for a HermitianObs.
     *
     * @param obs A HermitianObs object.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Pennylane::LightningKokkos::Observables::HermitianObs<
                StateVectorT> &obs) -> PrecisionT {
        return expval(obs.getMatrix(), obs.getWires());
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<ComplexT> &matrix_,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        PL_ABORT_IF(matrix_.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        return getExpValMatrix(vector2view(matrix_), wires);
    };

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::string &operation,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        switch (expval_funcs_[operation]) {
        case ExpValFunc::Identity:
            return applyExpValNamedFunctor<getExpectationValueIdentityFunctor,
                                           0>(wires);
        case ExpValFunc::PauliX:
            return applyExpValNamedFunctor<getExpectationValuePauliXFunctor, 1>(
                wires);
        case ExpValFunc::PauliY:
            return applyExpValNamedFunctor<getExpectationValuePauliYFunctor, 1>(
                wires);
        case ExpValFunc::PauliZ:
            return applyExpValNamedFunctor<getExpectationValuePauliZFunctor, 1>(
                wires);
        case ExpValFunc::Hadamard:
            return applyExpValNamedFunctor<getExpectationValueHadamardFunctor,
                                           1>(wires);
        default:
            PL_ABORT(
                std::string("Expval does not exist for named observable ") +
                operation);
        }
    };

    /**
     * @brief Expected value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    auto expval(const std::vector<op_type> &operations_list,
                const std::vector<std::vector<std::size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> expected_value_list;

        for (std::size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs An Observable object.
     * @param num_shots Number of shots.
     * @param shot_range Vector of shot number to measurement.
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs,
                const std::size_t &num_shots,
                const std::vector<std::size_t> &shot_range) -> PrecisionT {
        return BaseType::expval(obs, num_shots, shot_range);
    }

    /**
     * @brief Expected value of a Sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point expected value of the observable.
     */
    template <class index_type>
    PrecisionT expval(const index_type *row_map_ptr,
                      const index_type row_map_size,
                      const index_type *entries_ptr, const ComplexT *values_ptr,
                      const index_type numNNZ) {
        const Kokkos::View<ComplexT *> arr_data = this->_statevector.getView();
        PrecisionT expval = 0.0;
        KokkosSizeTVector kok_row_map("row_map", row_map_size);
        KokkosSizeTVector kok_indices("indices", numNNZ);
        KokkosVector kok_data("data", numNNZ);

        Kokkos::deep_copy(kok_data,
                          UnmanagedConstComplexHostView(values_ptr, numNNZ));
        Kokkos::deep_copy(kok_indices,
                          UnmanagedConstSizeTHostView(entries_ptr, numNNZ));
        Kokkos::deep_copy(kok_row_map, UnmanagedConstSizeTHostView(
                                           row_map_ptr, row_map_size));

        Kokkos::parallel_reduce(
            row_map_size - 1,
            getExpectationValueSparseFunctor<PrecisionT>(
                arr_data, kok_data, kok_indices, kok_row_map),
            expval);
        return expval;
    };

    /**
     * @brief Calculate variance of a general Observable.
     *
     * @param obs An Observable object.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        StateVectorT ob_sv{this->_statevector};
        obs.applyInPlace(ob_sv);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::string &operation,
             const std::vector<std::size_t> &wires) -> PrecisionT {
        StateVectorT ob_sv{this->_statevector};
        ob_sv.applyOperation(operation, wires);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of a Hermitian matrix.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::vector<ComplexT> &matrix,
             const std::vector<std::size_t> &wires) -> PrecisionT {
        StateVectorT ob_sv{this->_statevector};
        ob_sv.applyMatrix(matrix, wires);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the
     observables.
     */
    template <typename op_type>
    auto var(const std::vector<op_type> &operations_list,
             const std::vector<std::vector<std::size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<PrecisionT> expected_value_list;

        for (std::size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of a sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point with the variance of the sparse Hamiltonian.
     */
    template <class index_type>
    auto var(const index_type *row_map_ptr, const index_type row_map_size,
             const index_type *entries_ptr, const ComplexT *values_ptr,
             const index_type numNNZ) -> PrecisionT {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (std::size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");

        StateVectorT ob_sv{this->_statevector};

        SparseMV_Kokkos<PrecisionT>(this->_statevector.getView(),
                                    ob_sv.getView(), row_map_ptr, row_map_size,
                                    entries_ptr, values_ptr, numNNZ);

        const PrecisionT mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT squared_mean = static_cast<PrecisionT>(
            std::pow(getRealOfComplexInnerProduct(this->_statevector.getView(),
                                                  ob_sv.getView()),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Calculate the variance for an observable with the number of shots.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots.
     *
     * @return Variance of the given observable.
     */

    auto var(const Observable<StateVectorT> &obs, const std::size_t &num_shots)
        -> PrecisionT {
        return BaseType::var(obs, num_shots);
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @return Kokkos::View with probabilities
     * in lexicographic order.
     */
    auto probs_core() -> Kokkos::View<PrecisionT *> {
        const std::size_t N = this->_statevector.getLength();
        auto sv = this->_statevector.getView();
        Kokkos::View<PrecisionT *> d_probs("d_probs", N);
        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            KOKKOS_LAMBDA(const std::size_t k) {
                const PrecisionT rsv = sv(k).real();
                const PrecisionT isv = sv(k).imag();
                d_probs(k) = rsv * rsv + isv * isv;
            });
        return d_probs;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        return view2vector(probs_core());
    }

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    auto
    probs(const std::vector<std::size_t> &wires,
          [[maybe_unused]] const std::vector<std::size_t> &device_wires = {})
        -> std::vector<PrecisionT> {
        // GPU_SHARED_NWIRES_MAX is an upper bound for the size of the GPU array
        // used to reduce the probs (max size = 2 ** 7)
        constexpr std::size_t GPU_SHARED_NWIRES_MAX = 7;
        // BITSHIFT_FREE_WIRES_MIN is a lower bound for the size of the loop
        // over which the probs computation is parallelized in
        // `probs_bitshift_generic` The free wires are the wires which are
        // summed over.
        constexpr std::size_t BITSHIFT_FREE_WIRES_MIN = 10;
        // BITSHIFT_NWIRES_MAX is an upper bound for using
        // `probs_bitshift_generic`, beyond that size the other implementation
        // si more efficient
        constexpr std::size_t BITSHIFT_NWIRES_MAX = 9;
        // MDRANGE_NWIRES_MAX is an upper bound for using `MDRangePolicy` to
        // parallelize the probs computation. Beyond that size, parallelizing
        // over the probs elements is more efficient.
        constexpr std::size_t MDRANGE_NWIRES_MAX = 20;
        const std::size_t n_wires = wires.size();
        if (n_wires == 0) {
            return {1.0};
        }
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        // is_equal_to_all_wires is True if `wires` includes all wires in order
        // and false otherwise
        bool is_equal_to_all_wires = n_wires == num_qubits;
        for (std::size_t k = 0; k < n_wires; k++) {
            if (!is_equal_to_all_wires) {
                break;
            }
            is_equal_to_all_wires = wires[k] == k;
        }
        if (is_equal_to_all_wires) {
            return probs();
        }
        const bool is_gpu_scratch_limited =
            n_wires > GPU_SHARED_NWIRES_MAX &&
            !std::is_same_v<KokkosExecSpace, HostExecSpace>;
        if (num_qubits - n_wires > BITSHIFT_FREE_WIRES_MIN &&
            n_wires < BITSHIFT_NWIRES_MAX && !is_gpu_scratch_limited) {
            return probs_bitshift_generic<KokkosExecSpace>(
                this->_statevector.getView(), num_qubits, wires);
        }
        std::vector<std::size_t> all_indices =
            Pennylane::Util::generateBitsPatterns(wires, num_qubits);
        Kokkos::View<std::size_t *> d_all_indices = vector2view(all_indices);
        std::vector<std::size_t> all_offsets =
            Pennylane::Util::generateBitsPatterns(
                Pennylane::Util::getIndicesAfterExclusion(wires, num_qubits),
                num_qubits);
        Kokkos::View<std::size_t *> d_all_offsets = vector2view(all_offsets);
        Kokkos::View<PrecisionT *> d_probabilities("d_probabilities",
                                                   all_indices.size());
        Kokkos::View<ComplexT *> sv = this->_statevector.getView();

        // Reducing over `d_probabilities` requires too much L0 scratch memory
        // on GPUs. If n_wires >= 20, this also requires quite a bit of memory
        // on CPUs, so we fallback to the next implementation
        if (n_wires < MDRANGE_NWIRES_MAX && !is_gpu_scratch_limited) {
            using MDPolicyType_2D =
                Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left>>;
            auto md_policy = MDPolicyType_2D(
                {{0, 0}}, {{static_cast<int64_t>(all_indices.size()),
                            static_cast<int64_t>(all_offsets.size())}});
            Kokkos::parallel_reduce(
                md_policy,
                getProbsFunctor<PrecisionT, KokkosExecSpace>(
                    sv, wires, d_all_indices, d_all_offsets),
                d_probabilities);
        } else {
            Kokkos::parallel_for(
                all_indices.size(), KOKKOS_LAMBDA(const std::size_t i) {
                    for (std::size_t j = 0; j < d_all_offsets.size(); j++) {
                        const std::size_t index =
                            d_all_indices(i) + d_all_offsets(j);
                        const PrecisionT rsv = sv(index).real();
                        const PrecisionT isv = sv(index).imag();
                        d_probabilities(i) += rsv * rsv + isv * isv;
                    }
                });
        }

        return view2vector(d_probabilities);
    }

    /**
     * @brief Probabilities of each computational basis state for an observable.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots. If specified with a non-zero number,
     * shot-noise will be added to return probabilities
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs(const Observable<StateVectorT> &obs, std::size_t num_shots = 0)
        -> std::vector<PrecisionT> {
        return BaseType::probs(obs, num_shots);
    }

    /**
     * @brief Probabilities with shot-noise.
     *
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */
    auto probs(std::size_t num_shots) -> std::vector<PrecisionT> {
        return BaseType::probs(num_shots);
    }

    /**
     * @brief Probabilities with shot-noise for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @param num_shots Number of shots.
     *
     * @return Floating point std::vector with probabilities.
     */

    auto probs(const std::vector<std::size_t> &wires, std::size_t num_shots)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF_NOT(
            std::is_sorted(wires.cbegin(), wires.cend()),
            "LightningKokkos does not currently support out-of-order wire "
            "indices with probability calculations");

        return BaseType::probs(wires, num_shots);
    }

    /**
     * @brief  Inverse transform sampling method for samples.
     * Reference https://en.wikipedia.org/wiki/Inverse_transform_sampling
     *
     * @param num_samples Number of Samples
     * @param seed Seed to generate the samples from
     *
     * @return std::vector<std::size_t> to the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(std::size_t num_samples,
                          const std::optional<std::size_t> &seed = std::nullopt)
        -> std::vector<std::size_t> {
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        const std::size_t N = this->_statevector.getLength();
        Kokkos::View<std::size_t *> samples("num_samples",
                                            num_samples * num_qubits);

        // Convert probability distribution to cumulative distribution
        auto probability = probs_core();
        Kokkos::parallel_scan(
            Kokkos::RangePolicy<KokkosExecSpace>(0, N),
            KOKKOS_LAMBDA(const std::size_t k, PrecisionT &update_value,
                          const bool is_final) {
                const PrecisionT val_k = probability(k);
                if (is_final)
                    probability(k) = update_value;
                update_value += val_k;
            });

        // Sampling using Random_XorShift64_Pool
        auto rand_pool = seed.has_value()
                             ? Kokkos::Random_XorShift64_Pool<>(seed.value())
                             : Kokkos::Random_XorShift64_Pool<>(
                                   std::chrono::high_resolution_clock::now()
                                       .time_since_epoch()
                                       .count());

        Kokkos::parallel_for(
            Kokkos::RangePolicy<KokkosExecSpace>(0, num_samples),
            Sampler<PrecisionT, Kokkos::Random_XorShift64_Pool>(
                samples, probability, rand_pool, num_qubits, N));

        return view2vector(samples);
    }

  private:
    std::unordered_map<std::string, ExpValFunc> expval_funcs_;

    // clang-format off
    /**
    * @brief Register generator operations in the generators_indices_ attribute:
    *        an unordered_map mapping strings to GateOperation enumeration keywords.
    */
    void init_expval_funcs_() {
        expval_funcs_["Identity"] = ExpValFunc::Identity;
        expval_funcs_["PauliX"]   = ExpValFunc::PauliX;
        expval_funcs_["PauliY"]   = ExpValFunc::PauliY;
        expval_funcs_["PauliZ"]   = ExpValFunc::PauliZ;
        expval_funcs_["Hadamard"] = ExpValFunc::Hadamard;
    }
    // clang-format on
};

} // namespace Pennylane::LightningKokkos::Measures
