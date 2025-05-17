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
#include <iostream>

#include "ExpValFunctors.hpp"
#include "LinearAlgebraKokkos.hpp" // getRealOfComplexInnerProduct
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

  public:
    explicit MeasurementsMPI(StateVectorT &statevector)
        : BaseType{statevector} {};
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

        // if (operation == "Identity") {
        //     return 1.0;
        // }

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
        this->_statevector.barrier();
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

            // DO SWAP HERE
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

            std::size_t global_z_mask = 0;
            std::size_t global_index =
                this->_statevector.getGlobalIndexFromMPIRank(
                    this->_statevector.getMPIManager().getRank());
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
            /* std::cout << "global wires = ";
            for (auto const& wire : this->_statevector.getGlobalWires()) {
                std::cout << wire << " ";
            }
            std::cout << std::endl;
            std::cout << "local wires = ";
            for (auto const& wire : this->_statevector.getLocalWires()) {
                std::cout << wire << " ";
            }
            std::cout << "global index MPI map = ";
            for (auto const& wire :
            this->_statevector.getMPIRankToGlobalIndexMap()) { std::cout << wire
            << " ";
            }
            std::cout << " Local pauli word = " << local_pauli_word << " and
            local target wires = "; for (const auto& wire: local_target_wires) {
                std::cout << wire << " ";
            }
            std::cout          << " and converted to local wire indices = " ;
            for (const auto& wire:
            this->_statevector.getLocalWireIndices(local_target_wires)) {
                std::cout << wire << " ";
            }
            std::cout << "I am rank " << this->_statevector.getMPIRank() << "
            and global index = " <<
            this->_statevector.getGlobalIndexFromMPIRank(this->_statevector.getMPIRank())
                      << " and BEFORE multiplying -1s my local expval = "
                      << local_expval << std::endl;
 */

            if (std::popcount(global_index & global_z_mask) % 2 == 1) {
                local_expval *= -1.0;
            }
            /* std::cout << "I am rank " << this->_statevector.getMPIRank() << "
               and global index = " <<
               this->_statevector.getGlobalIndexFromMPIRank(this->_statevector.getMPIRank())
                      << " and AFTER multiplying -1s my local expval = "
                      << local_expval << std::endl;
 */
            // combine
            PrecisionT global_expval = 0.0;
            global_expval = this->_statevector.allReduceSum(local_expval);
            this->_statevector.barrier();
            result += global_expval * coeffs[word];
        }

        return result;
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
};

auto generate_samples(std::size_t num_samples) -> std::vector<std::size_t> {
    PL_ABORT("Generate Samples not supported in LK-MPI");
    std::vector<std::size_t> samples(num_samples);
    return samples;
}

} // namespace Pennylane::LightningKokkos::Measures
