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

#include <complex>
#include <stdio.h>
#include <vector>

#include "StateVector.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using std::complex;
using std::size_t;
using std::vector;
}; // namespace
/// @endcond

namespace Pennylane {
using namespace Util;
/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix) or through a
 * string-based function dispatch.
 *
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <class fp_t = double> class Measures {
  private:
    StateVector<fp_t> measured_statevector;

  public:
    Measures(StateVector<fp_t> provided_statevector)
        : measured_statevector{provided_statevector} {};

    /**
     * @brief Probabilities of each computational basis state.
     *
     * @return Floating point std::vector with probabilities
     * in lexicographic order according to wires.
     */
    vector<fp_t> probs() {
        complex<fp_t> *arr_data = measured_statevector.getData();
        vector<double> basis_probs(measured_statevector.getLength(), 0);
        for (size_t ind = 0; ind < 8; ind++) {
            basis_probs[ind] = std::pow(std::abs(arr_data[ind]), 2.0);
        }
        return basis_probs;
    };

    /**
     * @brief Probabilities for a subset of the full system.
     *
     * @param wires Wires will restrict probabilities to a subset of the full
     * system.
     * @return Floating point std::vector with probabilities
     * in lexicographic order according to wires.
     */
    vector<fp_t> probs(const vector<size_t> &wires) {
        complex<fp_t> *arr_data = measured_statevector.getData();
        vector<double> basis_probs(measured_statevector.getLength(), 0);
        for (size_t ind = 0; ind < 8; ind++) {
            basis_probs[ind] = std::pow(std::abs(arr_data[ind]), 2.0);
        }

        vector<size_t> all_indices =
            measured_statevector.generateBitPatterns(wires);
        vector<size_t> all_offsets = measured_statevector.generateBitPatterns(
            measured_statevector.getIndicesAfterExclusion(wires));

        vector<fp_t> probabilities(all_indices.size(), 0);

        size_t ind_probs = 0;
        for (auto index : all_indices) {
            for (auto offset : all_offsets) {
                probabilities[ind_probs] += basis_probs[index + offset];
            }
            ind_probs++;
        }

        return probabilities;
    };

    /**
     * @brief Expected value of an observable.
     *
     * @param operation Square matrix in row-major order or string with the
     * operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    template <typename op_type>
    fp_t expval(const op_type &operation, const vector<size_t> &wires) {
        size_t data_size = measured_statevector.getLength();

        // Copying the original state vector, for the application of the
        // observable operator.
        complex<double> *original_data;
        original_data = new complex<double>[data_size];
        std::copy(measured_statevector.getData(),
                  measured_statevector.getData() + data_size, original_data);
        StateVector<double> operator_statevector(original_data, data_size);

        operator_statevector.applyOperation(operation, wires);

        complex<fp_t> expected_value =
            innerProdC(measured_statevector.getData(),
                       operator_statevector.getData(), data_size);
        return std::real(expected_value);
    };

    /**
     * @brief Expected value for a list of observables.
     *
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    template <typename op_type>
    vector<fp_t> expval(const vector<op_type> &operations_list,
                        const vector<vector<size_t>> &wires_list) {
        if (operations_list.size() != wires_list.size()) {
            throw std::out_of_range("The lengths of the list of operations and "
                                    "wires do not match.");
        }

        vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of an observable.
     *
     * @param operation Square matrix in row-major order or string with the
     * operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observables.
     */
    template <typename op_type>
    fp_t var(const op_type &operation, const vector<size_t> &wires) {
        size_t data_size = measured_statevector.getLength();
        // Copying the original state vector, for the application of the
        // observable operator.
        complex<double> *original_data;
        original_data = new complex<double>[data_size];
        std::copy(measured_statevector.getData(),
                  measured_statevector.getData() + data_size, original_data);
        StateVector<double> operator_statevector(original_data, data_size);

        measured_statevector.applyOperation(operation, wires);

        fp_t mean_square =
            std::real(innerProdC(measured_statevector.getData(),
                                 measured_statevector.getData(), data_size));
        fp_t squared_mean =
            std::real(innerProdC(operator_statevector.getData(),
                                 measured_statevector.getData(), data_size));
        squared_mean = std::pow(squared_mean, 2);
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
    vector<fp_t> var(const vector<op_type> &operations_list,
                     const vector<vector<size_t>> &wires_list) {
        if (operations_list.size() != wires_list.size()) {
            throw std::out_of_range("The lengths of the list of operations and "
                                    "wires do not match.");
        }

        vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };
}; // class Measures

} // namespace Pennylane