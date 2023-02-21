// Copyright 2022 Xanadu Quantum Technologies Inc.

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
#include <random>
#include <stack>
#include <unordered_map>
#include <vector>

#include "Kokkos_Sparse.hpp"
#include "LinearAlgebra.hpp"
#include "Observables.hpp"
#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"

namespace Pennylane::Simulators {

/**
 * @brief Observable's Measurement Class.
 *
 * This class performs measurements in the state vector provided to its
 * constructor. Observables are defined by its operator(matrix) or through a
 * string-based function dispatch.
 *
 * @tparam fp_t Floating point precision of underlying measurements.
 * @tparam SVType type of the statevector to be measured.
 */
template <class fp_t = double, class SVType = StateVectorRawCPU<fp_t>>
class Measures {
  private:
    const SVType &original_statevector;
    using CFP_t = std::complex<fp_t>;

  public:
    explicit Measures(const SVType &provided_statevector)
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
    std::vector<fp_t>
    probs(const std::vector<size_t> &wires,
          [[maybe_unused]] const std::vector<size_t> &device_wires = {}) {
        // Determining index that would sort the vector.
        // This information is needed later.
        const auto sorted_ind_wires = Util::sorting_indices(wires);
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
        StateVectorManagedCPU<fp_t> operator_statevector(original_statevector);

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
        StateVectorManagedCPU<fp_t> operator_statevector(original_statevector);

        operator_statevector.applyOperation(operation, wires);

        CFP_t expected_value = Util::innerProdC(
            original_statevector.getData(), operator_statevector.getData(),
            original_statevector.getLength());
        return std::real(expected_value);
    };

    /**
     * @brief Expected value of a Sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return fp_t floating point precision.
     */
    template <class index_type>
    fp_t expval(const index_type *row_map_ptr, const index_type row_map_size,
                const index_type *entries_ptr, const CFP_t *values_ptr,
                const index_type numNNZ) {
        PL_ABORT_IF(
            (original_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            original_statevector.getData(),
            static_cast<index_type>(original_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        CFP_t expected_value = Util::innerProdC(
            original_statevector.getData(), operator_vector.data(),
            original_statevector.getLength());
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
    std::vector<fp_t>
    expval(const std::vector<op_type> &operations_list,
           const std::vector<std::vector<size_t>> &wires_list) {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Expectation value for a general Observable
     *
     * @param ob Observable
     */
    auto expval(const Observable<fp_t> &ob) -> fp_t {
        StateVectorManagedCPU<fp_t> op_sv(original_statevector);
        ob.applyInPlace(op_sv);
        const auto inner_prod =
            Util::innerProdC(original_statevector.getData(), op_sv.getData(),
                             original_statevector.getLength());
        return std::real(inner_prod);
    }

    /**
     * @brief Variance value for a general Observable
     *
     * @param ob Observable
     */
    auto var(const Observable<fp_t> &ob) -> fp_t {
        // Copying the original state vector, for the application of the
        // observable operator.
        StateVectorManagedCPU<fp_t> op_sv(original_statevector);
        ob.applyInPlace(op_sv);

        const fp_t mean_square = std::real(Util::innerProdC(
            op_sv.getData(), op_sv.getData(), op_sv.getLength()));
        const fp_t squared_mean = static_cast<fp_t>(std::pow(
            std::real(Util::innerProdC(original_statevector.getData(),
                                       op_sv.getData(), op_sv.getLength())),
            2));
        return (mean_square - squared_mean);
    }

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
        StateVectorManagedCPU<fp_t> operator_statevector(original_statevector);

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
        StateVectorManagedCPU<fp_t> operator_statevector(original_statevector);

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
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the observables.
     */
    template <typename op_type>
    std::vector<fp_t> var(const std::vector<op_type> &operations_list,
                          const std::vector<std::vector<size_t>> &wires_list) {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<fp_t> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of a Sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return fp_t Variance value.
     */
    template <class index_type>
    fp_t var(const index_type *row_map_ptr, const index_type row_map_size,
             const index_type *entries_ptr, const CFP_t *values_ptr,
             const index_type numNNZ) {
        PL_ABORT_IF(
            (original_statevector.getLength() != (size_t(row_map_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");
        auto operator_vector = Util::apply_Sparse_Matrix(
            original_statevector.getData(),
            static_cast<index_type>(original_statevector.getLength()),
            row_map_ptr, row_map_size, entries_ptr, values_ptr, numNNZ);

        const fp_t mean_square = std::real(
            Util::innerProdC(operator_vector.data(), operator_vector.data(),
                             operator_vector.size()));
        const fp_t squared_mean = static_cast<fp_t>(
            std::pow(std::real(Util::innerProdC(operator_vector.data(),
                                                original_statevector.getData(),
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
    std::vector<size_t> generate_samples(size_t num_samples) {
        const size_t num_qubits = original_statevector.getNumQubits();
        auto &&probabilities = probs();

        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::mt19937 generator(std::random_device{}());
        std::uniform_real_distribution<fp_t> distribution(0.0, 1.0);
        std::unordered_map<size_t, size_t> cache;

        const size_t N = probabilities.size();
        std::vector<double> bucket(N);
        std::vector<size_t> bucket_partner(N);
        std::stack<size_t> overfull_bucket_ids;
        std::stack<size_t> underfull_bucket_ids;

        for (size_t i = 0; i < N; i++) {
            bucket[i] = N * probabilities[i];
            bucket_partner[i] = i;
            if (bucket[i] > 1.0) {
                overfull_bucket_ids.push(i);
            }
            if (bucket[i] < 1.0) {
                underfull_bucket_ids.push(i);
            }
        }

        // Run alias algorithm
        while (!underfull_bucket_ids.empty() && !overfull_bucket_ids.empty()) {
            // get an overfull bucket
            size_t i = overfull_bucket_ids.top();

            // get an underfull bucket
            size_t j = underfull_bucket_ids.top();
            underfull_bucket_ids.pop();

            // underfull bucket is partned with an overfull bucket
            bucket_partner[j] = i;
            bucket[i] = bucket[i] + bucket[j] - 1;

            // if overfull bucket is now underfull
            // put in underfull stack
            if (bucket[i] < 1) {
                overfull_bucket_ids.pop();
                underfull_bucket_ids.push(i);
            }

            // if overfull bucket is full -> remove
            else if (bucket[i] == 1.0) {
                overfull_bucket_ids.pop();
            }
        }

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            fp_t pct = distribution(generator) * N;
            auto idx = static_cast<size_t>(pct);
            if (pct - idx > bucket[idx]) {
                idx = bucket_partner[idx];
            }
            // If cached, retrieve sample from cache
            if (cache.contains(idx)) {
                size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }
            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }
        return samples;
    }
}; // class Measures
} // namespace Pennylane::Simulators
