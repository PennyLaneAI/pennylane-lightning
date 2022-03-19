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
 * @file Define functions to compute the vector-Jacobian products
 * following the implementation in Pennylane.
 */
#pragma once

#include <algorithm>

#include "AdjointDiff.hpp"
#include "JacobianTape.hpp"
#include "LinearAlgebra.hpp"

namespace Pennylane::Algorithms {
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
template <typename T>
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
 * TODO: Change pointers to std::span in C++20
 */
template <typename PrecisionT>
void expvalVJP(const JacobianData<PrecisionT>& jd, const PrecisionT* dy,
               const PrecisionT* v_out, bool apply_operations = false) {

    const size_t num_params = jd.getTrainableParams().size();

    if (num_params == 0) {
        return ;
    }

    /*
    std::vector<PrecisionT> jac(jd.getNumObservables() * num_params, 0);

    // Compute Jacobian for the input jd using `adjoint` method
    adjointJacobian(jac, jd, apply_operations);

    // Compute VJP
    computeVJP(vjp, jac, dy, jd.getNumObservables(), num_params);
    */
}
} // namespace Pennylane::Algorithms
