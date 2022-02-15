// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * Defines a class for the measurement of observables
 * in quantum states represented by the StateVector class.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <vector>

#include "LinearAlgebra.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"

namespace Pennylane {
/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix) or through a
 * string-based function dispatch.
 *
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <class fp_t = double, class SVType = StateVectorRaw<fp_t>>
class Measures {
  private:
    const SVType &original_statevector;
    using CFP_t = std::complex<fp_t>;

  public:
    Measures(const SVType &provided_statevector)
        : original_statevector{provided_statevector} {};

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order.
     */
    std::vector<fp_t> probs() {
        const CFP_t *arr_data = original_statevector.getData();
        std::vector<fp_t> basis_probs(original_statevector.getLength(), 0);

        std::transform(arr_data, arr_data + original_statevector.getLength(),
                       basis_probs.begin(),
                       [](const CFP_t &z) -> fp_t { return std::norm(z); });

        return basis_probs;
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset
     * of the full system.
     * @return Floating point std::vector with probabilities.
     * The basis columns are rearranged according to wires.
     */
    std::vector<fp_t> probs(const std::vector<size_t> &wires) {
        // Determining index that would sort the vector.
        // This information is needed later.
        const std::vector<size_t> sorted_ind_wires(
            Util::sorting_indices(wires));
        // Sorting wires.
        std::vector<size_t> sorted_wires(wires.size());
        for (size_t pos = 0; pos < wires.size(); pos++) {
            sorted_wires[pos] = wires[sorted_ind_wires[pos]];
        }
        // Determining probabilities for the sorted wires.
        const CFP_t *arr_data = original_statevector.getData();

        size_t num_qubits = original_statevector.getNumQubits();

        const std::vector<size_t> all_indices =
            Gates::generateBitPatterns(sorted_wires, num_qubits);
        const std::vector<size_t> all_offsets = Gates::generateBitPatterns(
            Gates::getIndicesAfterExclusion(sorted_wires, num_qubits),
            num_qubits);

        std::vector<fp_t> probabilities(all_indices.size(), 0);

        size_t ind_probs = 0;
        for (auto index : all_indices) {
            for (auto offset : all_offsets) {
                probabilities[ind_probs] += std::norm(arr_data[index + offset]);
            }
            ind_probs++;
        }
        // Transposing the probabilities tensor with the indices determined at
        // the beginning.
        if (wires != sorted_wires) {
            probabilities =
                Util::transpose_state_tensor(probabilities, sorted_ind_wires);
        }
        return probabilities;
    }
    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    fp_t expval(const std::vector<CFP_t> &matrix,
                const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorManaged<fp_t> operator_statevector(original_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        CFP_t expected_value = Util::innerProdC(
            original_statevector.getData(), operator_statevector.getData(),
            original_statevector.getLength());
        return std::real(expected_value);
    };
    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    fp_t expval(const std::string &operation,
                const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorManaged<fp_t> operator_statevector(original_statevector);

        operator_statevector.applyOperation(operation, wires);

        CFP_t expected_value = Util::innerProdC(
            original_statevector.getData(), operator_statevector.getData(),
            original_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value for a list of observables.
     *
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    std::vector<fp_t>
    expval(const std::vector<op_type> &operations_list,
           const std::vector<std::vector<size_t>> &wires_list) {
        if (operations_list.size() != wires_list.size()) {
            throw std::out_of_range("The lengths of the list of operations and "
                                    "wires do not match.");
        }

        std::vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observables.
     */
    fp_t var(const std::string &operation, const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorManaged<fp_t> operator_statevector(original_statevector);

        operator_statevector.applyOperation(operation, wires);

        const std::complex<fp_t> *opsv_data = operator_statevector.getData();
        size_t orgsv_len = original_statevector.getLength();

        fp_t mean_square =
            std::real(Util::innerProdC(opsv_data, opsv_data, orgsv_len));
        fp_t squared_mean = std::real(Util::innerProdC(
            original_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<fp_t>(std::pow(squared_mean, 2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observables.
     */
    fp_t var(const std::vector<CFP_t> &matrix,
             const std::vector<size_t> &wires) {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorManaged<fp_t> operator_statevector(original_statevector);

        operator_statevector.applyMatrix(matrix, wires);

        const std::complex<fp_t> *opsv_data = operator_statevector.getData();
        size_t orgsv_len = original_statevector.getLength();

        fp_t mean_square =
            std::real(Util::innerProdC(opsv_data, opsv_data, orgsv_len));
        fp_t squared_mean = std::real(Util::innerProdC(
            original_statevector.getData(), opsv_data, orgsv_len));
        squared_mean = static_cast<fp_t>(std::pow(squared_mean, 2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the observables.
     */
    template <typename op_type>
    std::vector<fp_t> var(const std::vector<op_type> &operations_list,
                          const std::vector<std::vector<size_t>> &wires_list) {
        if (operations_list.size() != wires_list.size()) {
            throw std::out_of_range("The lengths of the list of operations and "
                                    "wires do not match.");
        }

        std::vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };
}; // class Measures
} // namespace Pennylane
