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
#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"

namespace Pennylane::Algorithms {

/**
 * @brief Represent the class to compute the vector-Jacobian products
 * following the implementation in Pennylane.
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class VectorJacobianProduct {
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
     * @param jac Row-wise flatten Jacobian matrix of shape `m * n`.
     * @param dy_row Gradient-output vector.
     * @param m Number of rows of `jac`.
     * @param n Number of columns of `jac`.
     */
    void computeVJP(std::vector<T> &vjp, const std::vector<T> &jac,
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
     * @param dy Gradient-output vector.
     * @param num_params Total number of parameters in the QuantumTape
     * @param apply_operations Indicate whether to apply operations to jd.psi
     * prior to calculation.
     *
     * @return std::function<std::vector<T>(const JacobianData<T> &jd)>
     * where `jd` is a JacobianData object representing the QuantumTape
     * to differentiate.
     *
     */
    auto vectorJacobianProduct(const std::vector<T> &dy, size_t num_params,
                               bool apply_operations = false)
        -> std::function<std::vector<T>(const JacobianData<T> &)> {
        if (dy.empty() ||
            std::all_of(dy.cbegin(), dy.cend(), [](T e) { return e == 0; })) {
            // If the dy vector is zero, then the
            // corresponding element of the VJP will be zero,
            // and we can avoid unnecessary computation.
            return
                [num_params =
                     num_params]([[maybe_unused]] const JacobianData<T> &jd)
                    -> std::vector<T> { return std::vector<T>(num_params, 0); };
        }

        return [=](const JacobianData<T> &jd) -> std::vector<T> {
            if (!jd.hasTrainableParams()) {
                // The jd has no trainable parameters;
                // the VJP is simple {}.
                return {};
            }

            std::vector<T> vjp(num_params);
            std::vector<T> jac(jd.getNumObservables() * num_params, 0);

            // Compute Jacobian for the input jd using `adjoint` method
            AdjointJacobian<T> v;
            v.adjointJacobian(jac, jd, apply_operations);

            // Compute VJP
            computeVJP(vjp, jac, dy, jd.getNumObservables(), num_params);
            return vjp;
        };
    }
}; // class VectorJacobianProduct

} // namespace Pennylane::Algorithms