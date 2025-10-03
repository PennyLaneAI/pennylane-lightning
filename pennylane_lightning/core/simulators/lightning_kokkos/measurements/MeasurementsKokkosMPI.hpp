// Copyright 2025 Xanadu Quantum Technologies Inc.

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

#include "ExpValFunctors.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
#include "MPIManagerKokkos.hpp"
#include "MeasurementsBase.hpp"
#include "MeasurementsKokkos.hpp"
#include "MeasuresFunctors.hpp"
#include "Observables.hpp"
#include "ObservablesKokkosMPI.hpp"
#include "StateVectorKokkos.hpp"
#include "Util.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

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
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Measures {
template <class StateVectorT>
class MeasurementsMPI final
    : public MeasurementsBase<StateVectorT, MeasurementsMPI<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType =
        MeasurementsBase<StateVectorT, MeasurementsMPI<StateVectorT>>;
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

    MPIManagerKokkos mpi_manager_;

  public:
    explicit MeasurementsMPI(StateVectorT &statevector)
        : BaseType{statevector}, mpi_manager_(statevector.getMPIManager()) {};
    /**
     * @brief Expectation value of an observable.
     *
     * @param sv Observable-state-vector product.
     * @return Floating point expectation value of the observable.
     */
    PrecisionT expval(StateVectorT &sv) {
        this->_statevector.matchWires(sv);

        const PrecisionT expected_value = getRealOfComplexInnerProduct(
            this->_statevector.getView(), sv.getView());
        return this->_statevector.allReduceSum(expected_value);
    }
    /**
     * @brief Expectation value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expectation value of the observable.
     */
    PrecisionT expval(const std::vector<ComplexT> &matrix,
                      const std::vector<std::size_t> &wires) {
        PL_ABORT_IF(matrix.size() != exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");

        if (!(this->_statevector.isWiresLocal(wires))) {
            auto global_wires_to_swap =
                this->_statevector.findGlobalWires(wires);
            auto local_wires_to_swap =
                this->_statevector.localWiresSubsetToSwap(global_wires_to_swap,
                                                          wires);
            this->_statevector.swapGlobalLocalWires(global_wires_to_swap,
                                                    local_wires_to_swap);
        }

        Measurements local_measure(this->_statevector.getLocalSV());
        PrecisionT local_expval = local_measure.expval(
            matrix, this->_statevector.getLocalWireIndices(wires));
        PrecisionT global_expval =
            this->_statevector.allReduceSum(local_expval);
        return global_expval;
    };

    /**
     * @brief Expectation value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expectation value of the observable.
     */
    PrecisionT expval(const std::string &operation,
                      const std::vector<size_t> &wires) {
        if (!(this->_statevector.isWiresLocal(wires))) {
            auto global_wires_to_swap =
                this->_statevector.findGlobalWires(wires);
            auto local_wires_to_swap =
                this->_statevector.localWiresSubsetToSwap(global_wires_to_swap,
                                                          wires);
            this->_statevector.swapGlobalLocalWires(global_wires_to_swap,
                                                    local_wires_to_swap);
        }

        Measurements local_measure(this->_statevector.getLocalSV());
        PrecisionT local_expval = local_measure.expval(
            operation, this->_statevector.getLocalWireIndices(wires));
        PrecisionT global_expval =
            this->_statevector.allReduceSum(local_expval);
        return global_expval;
    };

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param ob Observable.
     * @return Expectation value with respect to the given observable.
     */
    PrecisionT expval(Observable<StateVectorT> &ob) {
        StateVectorT ob_sv{this->_statevector};
        ob.applyInPlace(ob_sv);
        this->_statevector.matchWires(ob_sv);
        const PrecisionT expected_value = getRealOfComplexInnerProduct(
            this->_statevector.getView(), ob_sv.getView());
        const PrecisionT result =
            this->_statevector.allReduceSum(expected_value);
        return result;
    }

    /**
     * @brief Expectation value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expectation values for the
     * observables.
     */
    template <typename op_type>
    std::vector<PrecisionT>
    expval(const std::vector<op_type> &operations_list,
           const std::vector<std::vector<std::size_t>> &wires_list) {
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
     * @brief Expected value of a Pauli string (Pauli words with coefficients)
     *
     * @param pauli_words Vector of operators' name strings.
     * @param target_wires Vector of wires where to apply the operator.
     * @param coeffs Complex buffer of size |pauli_words|
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<std::string> &pauli_words,
                const std::vector<std::vector<std::size_t>> &target_wires,
                const std::vector<PrecisionT> &coeffs) -> PrecisionT {
        PrecisionT result = 0.0;
        for (std::size_t word = 0; word < pauli_words.size(); word++) {
            std::vector<std::size_t> X_wires;
            std::vector<std::size_t> Y_wires;
            std::vector<std::size_t> Z_wires;
            std::vector<std::size_t> global_wires_need_to_swap;
            std::vector<std::size_t> local_wires_cannot_be_swapped;
            std::vector<std::size_t> local_wires_to_swap;
            std::vector<std::size_t> local_target_wires;
            std::vector<std::size_t> global_Z_wires;
            std::string local_pauli_word{""};
            for (std::size_t i = 0; i < target_wires[word].size(); i++) {
                if (pauli_words[word][i] == 'X') {
                    X_wires.push_back(target_wires[word][i]);

                    this->_statevector.isWiresLocal({target_wires[word][i]})
                        ? local_wires_cannot_be_swapped.push_back(
                              target_wires[word][i])
                        : global_wires_need_to_swap.push_back(
                              target_wires[word][i]);

                    local_pauli_word += 'X';
                    local_target_wires.push_back(target_wires[word][i]);

                } else if (pauli_words[word][i] == 'Y') {
                    Y_wires.push_back(target_wires[word][i]);

                    this->_statevector.isWiresLocal({target_wires[word][i]})
                        ? local_wires_cannot_be_swapped.push_back(
                              target_wires[word][i])
                        : global_wires_need_to_swap.push_back(
                              target_wires[word][i]);

                    local_pauli_word += 'Y';
                    local_target_wires.push_back(target_wires[word][i]);

                } else if (pauli_words[word][i] == 'Z') {
                    Z_wires.push_back(target_wires[word][i]);
                }
            }
            PL_ABORT_IF((X_wires.size() + Y_wires.size() >
                         this->_statevector.getNumLocalWires()),
                        "Number of PauliX and PauliY in Pauli String exceeds "
                        "the number of local wires.");

            for (std::size_t i = 0; i < this->_statevector.getNumLocalWires();
                 i++) {
                if (local_wires_to_swap.size() ==
                    global_wires_need_to_swap.size()) {
                    break;
                }
                if (std::find(local_wires_cannot_be_swapped.begin(),
                              local_wires_cannot_be_swapped.end(),
                              this->_statevector.getLocalWires()[i]) ==
                    local_wires_cannot_be_swapped.end()) {
                    local_wires_to_swap.push_back(
                        this->_statevector.getLocalWires()[i]);
                }
            }

            // Do swap here
            this->_statevector.swapGlobalLocalWires(global_wires_need_to_swap,
                                                    local_wires_to_swap);
            // Construct local pauli string and local wires
            for (std::size_t z_wire : Z_wires) {
                if (this->_statevector.isWiresLocal({z_wire})) {
                    local_pauli_word += 'Z';
                    local_target_wires.push_back(z_wire);
                } else {
                    global_Z_wires.push_back(z_wire);
                }
            }

            // apply local expval
            Measurements local_measure(this->_statevector.getLocalSV());
            PrecisionT local_expval = local_measure.expval(
                {local_pauli_word},
                {this->_statevector.getLocalWireIndices(local_target_wires)},
                {1.0});

            // apply global expval
            std::size_t global_z_mask = 0;
            std::size_t global_index =
                this->_statevector.getGlobalIndexFromMPIRank(
                    mpi_manager_.getRank());
            for (std::size_t i = 0; i < global_Z_wires.size(); i++) {
                std::size_t distance = std::distance(
                    this->_statevector.getGlobalWires().begin(),
                    std::find(this->_statevector.getGlobalWires().begin(),
                              this->_statevector.getGlobalWires().end(),
                              global_Z_wires[i]));
                global_z_mask |=
                    (1U << (this->_statevector.getNumGlobalWires() - 1 -
                            distance));
            }

            if (std::popcount(global_index & global_z_mask) % 2 == 1) {
                local_expval *= -1.0;
            }
            // combine
            PrecisionT global_expval = 0.0;
            global_expval = this->_statevector.allReduceSum(local_expval);
            result += global_expval * coeffs[word];
        }

        return result;
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
     * @brief Variance of an observable.
     *
     * @param sv Observable-state-vector product.
     * @return Floating point variance of the observable.
     */
    PrecisionT var(StateVectorT &sv) {
        const PrecisionT local_mean_square =
            getRealOfComplexInnerProduct(sv.getView(), sv.getView());

        this->_statevector.matchWires(sv);
        const PrecisionT local_mean = getRealOfComplexInnerProduct(
            this->_statevector.getView(), sv.getView());
        const PrecisionT squared_mean =
            std::pow(this->_statevector.allReduceSum(local_mean), 2);

        const PrecisionT mean_square =
            this->_statevector.allReduceSum(local_mean_square);
        const PrecisionT variance = mean_square - squared_mean;
        return variance;
    }

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point variance of the observable.
     */
    PrecisionT var(const std::vector<ComplexT> &matrix,
                   const std::vector<std::size_t> &wires) {
        PrecisionT squared_mean = std::pow(expval(matrix, wires), 2);
        StateVectorT ob_sv{this->_statevector};
        ob_sv.applyOperation("Matrix", wires, false, {}, matrix);
        const PrecisionT local_mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT mean_square =
            this->_statevector.allReduceSum(local_mean_square);
        return mean_square - squared_mean;
    };

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point variance of the observable.
     */
    PrecisionT var(const std::string &operation,
                   const std::vector<size_t> &wires) {
        PrecisionT squared_mean = std::pow(expval(operation, wires), 2);
        StateVectorT ob_sv{this->_statevector};
        ob_sv.applyOperation(operation, wires, false, {});
        const PrecisionT local_mean_square =
            getRealOfComplexInnerProduct(ob_sv.getView(), ob_sv.getView());
        const PrecisionT mean_square =
            this->_statevector.allReduceSum(local_mean_square);
        return mean_square - squared_mean;
    };

    /**
     * @brief Calculate variance for a general Observable.
     *
     * @param ob Observable.
     * @return variance with respect to the given observable.
     */
    PrecisionT var(const Observable<StateVectorT> &ob) {
        StateVectorT ob_sv{this->_statevector};
        ob.applyInPlace(ob_sv);
        return var(ob_sv);
    }

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with variances for the
     * observables.
     */
    template <typename op_type>
    std::vector<PrecisionT>
    var(const std::vector<op_type> &operations_list,
        const std::vector<std::vector<std::size_t>> &wires_list) {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> variance_list;

        for (std::size_t index = 0; index < operations_list.size(); index++) {
            variance_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return variance_list;
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
     * @brief Probabilities for local state vector. Needs MPI Gather to collect
     * probabilities for the full state vector.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     *
     * This will return the probability local to the MPI rank/global index.
     * To obtain the full probability vector, you need to call
     * MPI Gather.
     */
    auto probs() -> std::vector<PrecisionT> {
        this->_statevector.reorderAllWires();
        auto local_meas = Measurements(this->_statevector.getLocalSV());
        auto local_probabilities = local_meas.probs();
        return local_probabilities;
    }

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @param device_wires Wires on the device.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     *
     * This will return the probability local to the MPI rank/global index.
     * To obtain the full probability vector, you need to call
     * MPI GatherV.
     */
    auto
    probs(const std::vector<std::size_t> &wires,
          [[maybe_unused]] const std::vector<std::size_t> &device_wires = {})
        -> std::vector<PrecisionT> {
        PL_ABORT_IF_NOT(
            std::is_sorted(wires.cbegin(), wires.cend()),
            "LightningKokkos does not currently support out-of-order wire "
            "indices with probability calculations");
        this->_statevector.reorderAllWires();

        auto global_wires = this->_statevector.findGlobalWires(wires);
        auto local_wires = this->_statevector.findLocalWires(wires);

        auto local_meas = Measurements(this->_statevector.getLocalSV());
        auto local_probabilities = local_meas.probs(
            this->_statevector.getLocalWireIndices(local_wires));

        std::size_t global_index = this->_statevector.getGlobalIndexFromMPIRank(
            mpi_manager_.getRank());
        std::size_t mask = 0;
        for (std::size_t i = 0; i < global_wires.size(); i++) {
            mask |= (1 << (this->_statevector.getRevGlobalWireIndex(
                         global_wires[i])));
        }
        std::size_t subCommGroupId = global_index & mask;

        auto sub_mpi_manager =
            mpi_manager_.split(subCommGroupId, mpi_manager_.getRank());

        if (sub_mpi_manager.getSize() == 1) {
            return local_probabilities;
        }
        std::vector<PrecisionT> subgroup_probabilities;
        if (sub_mpi_manager.getRank() == 0) {
            subgroup_probabilities.resize(exp2(local_wires.size()));
        }
        sub_mpi_manager.Reduce<PrecisionT>(local_probabilities,
                                           subgroup_probabilities, 0, "sum");
        return subgroup_probabilities;
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
     * @brief Utility method for samples.
     *
     * @param num_samples Number of samples to generate.
     *
     * Note that the generated samples are ordered by their global index (bin).
     */
    auto generate_samples(std::size_t num_samples) -> std::vector<std::size_t> {
        this->_statevector.reorderAllWires();
        auto local_sv_view = this->_statevector.getView();
        std::size_t num_global_qubits = this->_statevector.getNumGlobalWires();
        std::size_t num_local_qubits = this->_statevector.getNumLocalWires();
        std::size_t num_total_qubits = num_global_qubits + num_local_qubits;
        std::size_t twoN_global_qubits = exp2(num_global_qubits);
        std::size_t mpi_rank = mpi_manager_.getRank();
        std::size_t global_index =
            this->_statevector.getGlobalIndexFromMPIRank(mpi_rank);

        auto rand_pool =
            this->_deviceseed.has_value()
                ? Kokkos::Random_XorShift64_Pool<>(this->_deviceseed.value())
                : Kokkos::Random_XorShift64_Pool<>(
                      std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count());

        // Get local squared norm
        PrecisionT local_squared_norm = 0.0;
        Kokkos::parallel_reduce(
            local_sv_view.size(),
            KOKKOS_LAMBDA(std::size_t i, PrecisionT &sum) {
                const PrecisionT norm = Kokkos::abs(local_sv_view(i));
                sum += norm * norm;
            },
            local_squared_norm);
        Kokkos::fence();
        std::vector<PrecisionT> local_norms(mpi_manager_.getSize());

        mpi_manager_.Gather(local_squared_norm, local_norms, 0);

        // Decide how many samples to generate for each rank using total norm of
        // subSV
        Kokkos::View<std::size_t *> global_samples_bin("global_samples",
                                                       twoN_global_qubits);
        if (mpi_manager_.getRank() == 0) {
            auto d_local_squared_norm = vector2view(local_norms);

            Kokkos::parallel_scan(
                Kokkos::RangePolicy<KokkosExecSpace>(0, twoN_global_qubits),
                KOKKOS_LAMBDA(const std::size_t k, PrecisionT &update_value,
                              const bool is_final) {
                    const PrecisionT val_k = d_local_squared_norm(k);
                    if (is_final)
                        d_local_squared_norm(k) = update_value;
                    update_value += val_k;
                });
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, num_samples),
                Global_Bin_Sampler<PrecisionT, Kokkos::Random_XorShift64_Pool>(
                    global_samples_bin, d_local_squared_norm, rand_pool,
                    num_global_qubits, twoN_global_qubits));
            Kokkos::fence();
        }
        mpi_manager_.Barrier();
        mpi_manager_.Bcast(global_samples_bin, 0);
        std::vector<std::size_t> h_global_samples_bin(twoN_global_qubits);
        h_global_samples_bin = view2vector(global_samples_bin);

        std::size_t local_num_samples = h_global_samples_bin[mpi_rank];
        Kokkos::View<std::size_t *> local_samples(
            "local_num_samples",
            local_num_samples * (num_local_qubits + num_global_qubits));
        if (local_num_samples > 0) {
            // Normalized local probabilities
            auto local_normalized_probability =
                Measurements(this->_statevector.getLocalSV()).probs_core();
            const double inv_norm = 1. / local_squared_norm;
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(
                    0, local_normalized_probability.size()),
                KOKKOS_LAMBDA(const std::size_t k) {
                    local_normalized_probability(k) *= inv_norm;
                });

            // Convert probability distribution to cumulative distribution
            Kokkos::parallel_scan(
                Kokkos::RangePolicy<KokkosExecSpace>(0, exp2(num_local_qubits)),
                KOKKOS_LAMBDA(const std::size_t k, PrecisionT &update_value,
                              const bool is_final) {
                    const PrecisionT val_k = local_normalized_probability(k);
                    if (is_final)
                        local_normalized_probability(k) = update_value;
                    update_value += val_k;
                });

            // Generate local samples using local distribution
            Kokkos::parallel_for(
                Kokkos::RangePolicy<KokkosExecSpace>(0, local_num_samples),
                Local_Sampler<PrecisionT, Kokkos::Random_XorShift64_Pool>(
                    local_samples, local_normalized_probability, rand_pool,
                    num_local_qubits, num_global_qubits, global_index,
                    exp2(num_local_qubits)));

            Kokkos::fence();
        }

        Kokkos::View<std::size_t *> samples("num_samples",
                                            num_samples * num_total_qubits);

        std::vector<int> recv_counts(twoN_global_qubits);
        std::vector<int> displacements(twoN_global_qubits);
        for (std::size_t i = 0; i < recv_counts.size(); i++) {
            recv_counts[i] = h_global_samples_bin[i] * num_total_qubits;
        }
        displacements[0] = 0;
        for (std::size_t i = 1; i < recv_counts.size(); i++) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }

        mpi_manager_.AllGatherV(local_samples, samples, recv_counts,
                                displacements);
        return view2vector(samples);
    }
};
} // namespace Pennylane::LightningKokkos::Measures
