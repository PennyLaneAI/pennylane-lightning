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
 * @file
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Qubit StateVector class.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "ExpValFunc.hpp"
#include "LinearAlgebra.hpp"
#include "MeasurementKernels.hpp"
#include "MeasurementsBase.hpp"
#include "Observables.hpp"
#include "SparseLinAlg.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "StateVectorLQubitRaw.hpp"
#include "TransitionKernels.hpp"
#include "Util.hpp" //transpose_state_tensor, sorting_indices

/// @cond DEV
namespace {
using namespace Pennylane::Measures;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::Observables;
using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::LightningQubit::Util::innerProdC;
namespace PUtil = Pennylane::Util;
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

namespace Pennylane::LightningQubit::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class couples with a statevector to performs measurements.
 * Observables are defined by its operator(matrix), the observable class,
 * or through a string-based function dispatch.
 *
 * @tparam StateVectorT type of the statevector to be measured.
 */
template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {
  private:
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;

  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    explicit Measurements(const StateVectorT &statevector)
        : BaseType{statevector} {};

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    auto probs() -> std::vector<PrecisionT> {
        const ComplexT *arr_data = this->_statevector.getData();
        const std::size_t n_probs = this->_statevector.getLength();
        std::vector<PrecisionT> probabilities(n_probs);
        auto *probs = probabilities.data();
#if defined PL_LQ_KERNEL_OMP && defined _OPENMP
#pragma omp parallel for
#endif
        for (std::size_t k = 0; k < n_probs; k++) {
            probs[k] = std::norm(arr_data[k]);
        }
        return probabilities;
    };

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
            return this->probs();
        }

        const ComplexT *arr_data = this->_statevector.getData();

        // Templated 1-4 wire cases; return probs
        PROBS_SPECIAL_CASE(1);
        PROBS_SPECIAL_CASE(2);
        PROBS_SPECIAL_CASE(3);
        PROBS_SPECIAL_CASE(4);

        const std::vector<std::size_t> all_indices =
            Gates::generateBitPatterns(wires, num_qubits);
        const std::vector<std::size_t> all_offsets = Gates::generateBitPatterns(
            Gates::getIndicesAfterExclusion(wires, num_qubits), num_qubits);
        const std::size_t n_probs = PUtil::exp2(n_wires);
        std::vector<PrecisionT> probabilities(n_probs, 0);
        auto *probs = probabilities.data();
        // For 5 wires and more, there are at least 32 probs entries to
        // parallelize over This scheme was found most favorable in terms of
        // memory accesses and it prevents the stack overflow caused by
        // `reduction(+ : probs[ : n_probs])` when n_probs approaches 2**20
#if defined PL_LQ_KERNEL_OMP && defined _OPENMP
#pragma omp parallel for
#endif
        for (std::size_t ind_probs = 0; ind_probs < n_probs; ind_probs++) {
            for (auto offset : all_offsets) {
                probs[ind_probs] +=
                    std::norm(arr_data[all_indices[ind_probs] + offset]);
            }
        }
        return probabilities;
    }

    /**
     * @brief Probabilities to measure rotated basis states.
     *
     * @param obs An observable object.
     * @param num_shots Number of shots (Optional). If specified with a non-zero
     * number, shot-noise will be added to return probabilities
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
     * @param num_shots Number of shots.
     * of the full system.
     *
     * @return Floating point std::vector with probabilities.
     */

    auto probs(const std::vector<std::size_t> &wires, std::size_t num_shots)
        -> std::vector<PrecisionT> {
        return BaseType::probs(wires, num_shots);
    }

    /**
     * @brief Expected value of a matrix.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<ComplexT> &matrix,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        PL_ABORT_IF(matrix.size() != PUtil::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        const ComplexT *arr_data = this->_statevector.getData();

        switch (wires.size()) {
        case 1: {
            return applyExpValMatWires1<PrecisionT>(arr_data, num_qubits, wires,
                                                    matrix);
        }
        case 2: {
            return applyExpValMatWires2<PrecisionT>(arr_data, num_qubits, wires,
                                                    matrix);
        }
        case 3: {
            return applyExpValMatWires3<PrecisionT>(arr_data, num_qubits, wires,
                                                    matrix);
        }
        default:
            return applyExpValMatMultiQubit<PrecisionT>(arr_data, num_qubits,
                                                        wires, matrix);
        }
    };

    /**
     * @brief Expected value of a named observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires to apply the operator.
     * @return Expected value of the observable.
     */
    auto expval(const std::string &operation,
                const std::vector<std::size_t> &wires) -> PrecisionT {
        const ComplexT *arr_data = this->_statevector.getData();
        const std::size_t num_qubits = this->_statevector.getNumQubits();

        using FuncT = std::function<void(const ComplexT *, const std::size_t,
                                         const std::size_t, PrecisionT &)>;
        FuncT core_function;

        switch (expval_funcs_[operation]) {
        case ExpValFunc::Identity:
            return 1.0;
        case ExpValFunc::PauliX:
            core_function = [](const ComplexT *arr, const std::size_t i0,
                               const std::size_t i1,
                               PrecisionT &expected_value) {
                expected_value += (arr[i0].real() * arr[i1].real() +
                                   arr[i0].imag() * arr[i1].imag()) *
                                  2.0;
            };
            break;
        case ExpValFunc::PauliY:
            core_function = [](const ComplexT *arr, const std::size_t i0,
                               const std::size_t i1,
                               PrecisionT &expected_value) {
                expected_value += (arr[i0].real() * arr[i1].imag() -
                                   arr[i0].imag() * arr[i1].real()) *
                                  2.0;
            };
            break;
        case ExpValFunc::PauliZ:
            core_function = [](const ComplexT *arr, const std::size_t i0,
                               const std::size_t i1,
                               PrecisionT &expected_value) {
                expected_value += std::norm(arr[i0]) - std::norm(arr[i1]);
            };
            break;
        case ExpValFunc::Hadamard:
            core_function = [](const ComplexT *arr, const std::size_t i0,
                               const std::size_t i1,
                               PrecisionT &expected_value) {
                const ComplexT v0 = arr[i0];
                const ComplexT v1 = arr[i1];
                expected_value +=
                    M_SQRT1_2 *
                    (std::norm(v0) - std::norm(v1) +
                     2.0 * (v0.real() * v1.real() + v0.imag() * v1.imag()));
            };
            break;
        default:
            PL_ABORT(
                std::string("Expval does not exist for named observable ") +
                operation);
            break;
        }

        return applyExpVal1<PrecisionT, FuncT>(arr_data, num_qubits, wires,
                                               core_function);
    };

    /**
     * @brief Expected value of a Sparse Hamiltonian.
     *
     * @tparam IndexT integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param column_idx_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point expected value of the observable.
     */
    template <class IndexT>
    auto expval(const IndexT *row_map_ptr, const IndexT row_map_size,
                const IndexT *column_idx_ptr, const ComplexT *values_ptr,
                const IndexT numNNZ) -> PrecisionT {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (std::size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<IndexT>(this->_statevector.getLength()), row_map_ptr,
            row_map_size, column_idx_ptr, values_ptr, numNNZ);

        ComplexT expected_value =
            innerProdC(this->_statevector.getData(), operator_vector.data(),
                       this->_statevector.getLength());
        return std::real(expected_value);
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
     * @brief Expectation value for a general Observable
     *
     * @param obs An observable object.
     * @return Floating point expected value of the observable.
     */
    auto expval(const Observable<StateVectorT> &obs) -> PrecisionT {
        PrecisionT result{};

        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsExpval(sv, obs, this->_statevector);
        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsExpval(sv, obs, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }

        return result;
    }

    /**
     * @brief Expectation value for a Observable with shots
     *
     * @param obs An observable object.
     * @param num_shots Number of shots.
     * @param shot_range Vector of shot number to measurement
     * @return Floating point expected value of the observable.
     */

    auto expval(const Observable<StateVectorT> &obs,
                const std::size_t &num_shots,
                const std::vector<std::size_t> &shot_range) -> PrecisionT {
        return BaseType::expval(obs, num_shots, shot_range);
    }

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
     * @brief Variance value for a general Observable
     *
     * @param obs An observable object.
     * @return Floating point with the variance of the observable.
     */
    auto var(const Observable<StateVectorT> &obs) -> PrecisionT {
        PrecisionT result{};
        if constexpr (std::is_same_v<typename StateVectorT::MemoryStorageT,
                                     MemoryStorageLocation::Internal>) {
            StateVectorT sv(this->_statevector);
            result = calculateObsVar(sv, obs, this->_statevector);

        } else if constexpr (std::is_same_v<
                                 typename StateVectorT::MemoryStorageT,
                                 MemoryStorageLocation::External>) {
            std::vector<ComplexT> data_storage(
                this->_statevector.getData(),
                this->_statevector.getData() + this->_statevector.getLength());
            StateVectorT sv(data_storage.data(), data_storage.size());
            result = calculateObsVar(sv, obs, this->_statevector);
        } else {
            /// LCOV_EXCL_START
            PL_ABORT("Undefined memory storage location for StateVectorT.");
            /// LCOV_EXCL_STOP
        }
        return result;
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
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyOperation(operation, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        std::size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
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
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorLQubitManaged<PrecisionT> operator_statevector(
            this->_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        const std::complex<PrecisionT> *opsv_data =
            operator_statevector.getData();
        std::size_t orgsv_len = this->_statevector.getLength();

        PrecisionT mean_square =
            std::real(innerProdC(opsv_data, opsv_data, orgsv_len));
        PrecisionT squared_mean = std::real(
            innerProdC(this->_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<PrecisionT>(std::pow(squared_mean, 2));
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
     * @brief Generate samples using the Metropolis-Hastings method.
     * Reference: Numerical Recipes, NetKet paper
     *
     * @param transition_kernel User-defined functor for producing
     transitions
     * between metropolis states.
     * @param num_burnin Number of Metropolis burn-in steps.
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<std::size_t>
    generate_samples_metropolis(const std::string &kernelname,
                                std::size_t num_burnin,
                                std::size_t num_samples) {
        std::size_t num_qubits = this->_statevector.getNumQubits();
        std::uniform_real_distribution<PrecisionT> distrib(0.0, 1.0);
        std::vector<std::size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<std::size_t, std::size_t> cache;
        this->setSeed(this->_deviceseed);

        TransitionKernelType transition_kernel = TransitionKernelType::Local;
        if (kernelname == "NonZeroRandom") {
            transition_kernel = TransitionKernelType::NonZeroRandom;
        }

        auto tk =
            kernel_factory(transition_kernel, this->_statevector.getData(),
                           this->_statevector.getNumQubits(), this->_rng);
        std::size_t idx = 0;

        // Burn In
        for (std::size_t i = 0; i < num_burnin; i++) {
            idx = metropolis_step(this->_statevector, tk, this->_rng, distrib,
                                  idx); // Burn-in.
        }

        // Sample
        for (std::size_t i = 0; i < num_samples; i++) {
            idx = metropolis_step(this->_statevector, tk, this->_rng, distrib,
                                  idx);

            if (cache.contains(idx)) {
                std::size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }

            // If not cached, compute
            else {
                for (std::size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }
        return samples;
    }

    /**
     * @brief Variance of a sparse Hamiltonian.
     *
     * @tparam IndexT integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param column_idx_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point with the variance of the sparse Hamiltonian.
     */
    template <class IndexT>
    PrecisionT var(const IndexT *row_map_ptr, const IndexT row_map_size,
                   const IndexT *column_idx_ptr, const ComplexT *values_ptr,
                   const IndexT numNNZ) {
        PL_ABORT_IF(
            (this->_statevector.getLength() != (std::size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            this->_statevector.getData(),
            static_cast<IndexT>(this->_statevector.getLength()), row_map_ptr,
            row_map_size, column_idx_ptr, values_ptr, numNNZ);

        const PrecisionT mean_square =
            std::real(innerProdC(operator_vector.data(), operator_vector.data(),
                                 operator_vector.size()));
        const auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(operator_vector.data(),
                                          this->_statevector.getData(),
                                          operator_vector.size())),
                     2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Generate samples using the alias method.
     * Reference: https://en.wikipedia.org/wiki/Alias_method
     *
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<std::size_t> generate_samples(const std::size_t num_samples) {
        const std::size_t num_qubits = this->_statevector.getNumQubits();
        std::vector<std::size_t> wires(num_qubits);
        std::iota(wires.begin(), wires.end(), 0);
        return generate_samples(wires, num_samples);
    }

    /**
     * @brief Generate samples.
     *
     * @param wires Sample are generated for the specified wires.
     * @param num_samples The number of samples to generate.
     * @return 1-D vector of samples in binary, each sample is
     * separated by a stride equal to the number of qubits.
     */
    std::vector<std::size_t>
    generate_samples(const std::vector<std::size_t> &wires,
                     const std::size_t num_samples) {
        const std::size_t n_wires = wires.size();
        std::vector<std::size_t> samples(num_samples * n_wires);
        this->setSeed(this->_deviceseed);
        DiscreteRandomVariable<PrecisionT> drv{this->_rng, probs(wires)};
        // The Python layer expects a 2D array with dimensions (n_samples x
        // n_wires) and hence the linear index is `s * n_wires + (n_wires - 1 -
        // j)` `s` being the "slow" row index and `j` being the "fast" column
        // index
        for (std::size_t s = 0; s < num_samples; s++) {
            const std::size_t idx = drv();
            for (std::size_t j = 0; j < n_wires; j++) {
                samples[s * n_wires + (n_wires - 1 - j)] = (idx >> j) & 1U;
            }
        }
        return samples;
    }

  private:
    std::unordered_map<std::string, ExpValFunc> expval_funcs_ = {
        {"Identity", ExpValFunc::Identity},
        {"PauliX", ExpValFunc::PauliX},
        {"PauliY", ExpValFunc::PauliY},
        {"PauliZ", ExpValFunc::PauliZ},
        {"Hadamard", ExpValFunc::Hadamard}};
    /**
     * @brief Support function that calculates <bra|obs|ket> to obtain the
     * observable's expectation value.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsExpval(StateVectorT &bra,
                                   const Observable<StateVectorT> &obs,
                                   const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        return std::real(
            innerProdC(bra.getData(), ket.getData(), ket.getLength()));
    }

    /**
     * @brief Support function that calculates <bra|obs^2|ket> and
     * (<bra|obs|ket>)^2 to obtain the observable's variance.
     *
     * @param bra Reference to the statevector where the observable will be
     * applied, must be mutable.
     * @param obs Constant reference to an observable.
     * @param ket Constant reference to the base statevector.
     * @return PrecisionT
     */
    auto inline calculateObsVar(StateVectorT &bra,
                                const Observable<StateVectorT> &obs,
                                const StateVectorT &ket) -> PrecisionT {
        obs.applyInPlace(bra);
        PrecisionT mean_square = std::real(
            innerProdC(bra.getData(), bra.getData(), bra.getLength()));
        auto squared_mean = static_cast<PrecisionT>(
            std::pow(std::real(innerProdC(bra.getData(), ket.getData(),
                                          ket.getLength())),
                     2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Complete a single Metropolis-Hastings step.
     *
     * @param sv state vector
     * @param tk User-defined functor for producing transitions
     * between metropolis states.
     * @param gen Random number generator.
     * @param distrib Random number distribution.
     * @param init_idx Init index of basis state.
     */
    std::size_t
    metropolis_step(const StateVectorT &sv,
                    const std::unique_ptr<TransitionKernel<PrecisionT>> &tk,
                    std::mt19937 &gen,
                    std::uniform_real_distribution<PrecisionT> &distrib,
                    std::size_t init_idx) {
        auto init_plog = std::log(
            (sv.getData()[init_idx] * std::conj(sv.getData()[init_idx]))
                .real());

        auto init_qratio = tk->operator()(init_idx);

        // transition kernel outputs these two
        auto &trans_idx = init_qratio.first;
        auto &trans_qratio = init_qratio.second;

        auto trans_plog = std::log(
            (sv.getData()[trans_idx] * std::conj(sv.getData()[trans_idx]))
                .real());

        auto alph = std::min<PrecisionT>(
            1., trans_qratio * std::exp(trans_plog - init_plog));
        auto ran = distrib(gen);

        if (ran < alph) {
            return trans_idx;
        }
        return init_idx;
    }
}; // class Measurements
} // namespace Pennylane::LightningQubit::Measures
