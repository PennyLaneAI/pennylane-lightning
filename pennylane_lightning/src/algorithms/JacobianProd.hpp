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
#pragma once

#include <algorithm>

#include "AdjointDiff.hpp"

namespace Pennylane {
namespace Algorithms {

/**
 * @brief Represent the class to compute the vector-Jacobian products
 * following the implementation in Pennylane.
 *
 * @tparam T Floating-point precision.
 */
template <class T = double>
class VectorJacobianProduct : public AdjointJacobian<T> {
  private:
    /**
     * @brief Computes the vector-Jacobian product for a given vector of
     * gradient outputs and a Jacobian.
     *
     * @param res Prealloacted vector for row-major ordered `jac` matrix
     * representation.
     * @param jac Jacobian matrix from `AdjointJacobian`.
     * @param len Total allocation size of `jac`.
     */
    void getRowMajor(std::vector<T> &res,
                     const std::vector<std::vector<T>> &jac, size_t len = 0U) {
        if (jac.empty()) {
            return;
        }

        const size_t r_len = jac.size();
        const size_t c_len = jac.front().size();
        const size_t t_len = len != 0U ? len : r_len * c_len;

        if (res.size() != t_len) {
            res.resize(t_len);
        }

        size_t k = 0;
        for (size_t i = 0; i < r_len; i++) {
            for (size_t j = 0; j < c_len; j++) {
                res[k] = jac[i][j];
                k++;
            }
        }
    }

  public:
    VectorJacobianProduct() = default;

    /**
     * @brief Computes the vector-Jacobian product for a given vector of
     * gradient outputs and a Jacobian.
     *
     * @param vjp Preallocated vector for vector-jacobian product data results.
     * @param jac Jacobian matrix from `AdjointJacobian`.
     * @param dy_row Gradient-output vector.
     */
    void computeVJP(std::vector<T> &vjp, const std::vector<std::vector<T>> &jac,
                    const std::vector<T> &dy_row) {
        if (jac.empty() || dy_row.empty()) {
            vjp.clear();
            return;
        }

        const size_t r_len = jac.size();
        const size_t c_len = jac.front().size();
        if (dy_row.size() != r_len) {
            throw std::invalid_argument(
                "Invalid size for the gradient-output vector");
        }

        const size_t t_len = r_len * c_len;
        std::vector<T> jac_row(t_len);
        getRowMajor(jac_row, jac, t_len);

        Util::vecMatrixProd(vjp, dy_row, jac_row, r_len, c_len);
    }

    /**
     * @brief Computes the vector-Jacobian product for a given vector of
     * gradient outputs and a Jacobian.
     *
     * @param vjp Preallocated vector for vector-jacobian product data results.
     * @param jac Row-wise flatten Jacobian matrix of shape m * n.
     * @param dy_row Gradient-output vector.
     * @param m Number of rows of `jac`.
     * @param n Number of columns of `jac`.
     */
    void _computeVJP(std::vector<T> &vjp, const std::vector<T> &jac,
                     const std::vector<T> &dy_row, size_t m, size_t n) {
        if (jac.empty() || dy_row.empty()) {
            vjp.clear();
            return;
        }

        if (dy_row.size() != m) {
            throw std::invalid_argument(
                "Invalid size for the gradient-output vector");
        }

        Util::vecMatrixProd(vjp, dy_row, jac, m, n);
    }

    /**
     * @brief Calculates the VectorJacobianProduct for the statevector
     * for the selected set of parametric gates using `AdjointJacobian`.
     *
     * @param vjp Preallocated vector for vector-jacobian product data results
     * of size `trainableParams.size()`.
     * @param jac Preallocated Jacobian matrix from `AdjointJacobian` of size
     * `observables.size() * trainableParams.size()`.
     * @param psi Pointer to the statevector data.
     * @param num_elements Length of the statevector data.
     * @param dy Gradient-output vector.
     * @param observables Observables for which to calculate Jacobian.
     * @param operations Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void vectorJacobianProduct(std::vector<T> &vjp,
                               std::vector<std::vector<T>> &jac,
                               const std::vector<T> &dy,
                               const std::complex<T> *psi, size_t num_elements,
                               const std::vector<ObsDatum<T>> &observables,
                               const OpsData<T> &operations,
                               const std::vector<size_t> &trainableParams,
                               bool apply_operations = false) {
        const size_t num_params = trainableParams.size();

        if (num_params == 0U || dy.empty()) {
            vjp.clear();
            return;
        }

        const bool allzero =
            std::all_of(dy.cbegin(), dy.cend(), [](T e) { return e == 0; });
        if (allzero) {
            vjp.resize(num_params);
            return;
        }

        this->adjointJacobian(psi, num_elements, jac, observables, operations,
                              trainableParams, apply_operations);

        computeVJP(vjp, jac, dy);
    }
}; // class VectorJacobianProduct

} // namespace Algorithms
} // namespace Pennylane