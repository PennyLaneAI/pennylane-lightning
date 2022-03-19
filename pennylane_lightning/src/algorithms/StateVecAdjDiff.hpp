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
 */
#pragma once
#include "AlgUtil.hpp"

namespace Pennylane::Algorithms {
/**
 * @brief Compute vector Jacobian product for a statevector Jacobian.
 *
 * @rst
 * Product of statevector Jacobian :math:`J_{ij} = \partial_j \psi(i)` and
 * a vector. Note that :math:`J` is :math:`2^n \times m` matrix where
 * :math:`n` is the number of qubits and :math:`m` is the number of
 * trainable parameters in the tape.
 * Thus the result vector is length :math:`m`.
 * @endrst
 *
 * @param jd Jacobian data
 * @param vec A cotangent vector of size 2^n
 * @param apply_operations Assume the given state is an input state and apply
 * operations if true
 * @return a vector length of the number of trainable parameters
 *
 * TODO: change pointer parameters to std::span in C++20
 */
template <typename PrecisionT>
static void statevectorVJP(const JacobianData<PrecisionT> &jd,
                           const std::complex<PrecisionT> *dy,
                           const std::complex<PrecisionT> *vec_out,
                           bool apply_operations = false) {
    using ComplexPrecisionT = std::complex<PrecisionT>;
    if (!jd.hasTrainableParams()) {
        return ;
    }
    const OpsData<PrecisionT> &ops = jd.getOperations();
    const std::vector<std::string> &ops_name = ops.getOpsName();

    // We can assume the trainable params are sorted (from Python)
    const size_t num_param_ops = ops.getNumParOps();
    const auto trainable_params = jd.getTrainableParams();

    // Create $U_{1:p}\vert \lambda \rangle$
    StateVectorManaged<PrecisionT> lambda(jd.getPtrStateVec(),
                                          jd.getSizeStateVec());

    // Apply given operations to statevector if requested
    if (apply_operations) {
        applyOperations(lambda, ops);
    }
    StateVectorManaged<PrecisionT> mu(dy, jd.getSizeStateVec());
    StateVectorManaged<PrecisionT> mu_d(
        Util::log2PerfectPower(jd.getSizeStateVec()));

    const auto tp_rend = trainable_params.rend();
    auto tp_it = trainable_params.rbegin();
    size_t current_param_idx =
        num_param_ops - 1; // total number of parametric ops
    size_t trainable_param_idx = trainable_params.size() - 1;

    for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
         op_idx--) {
        PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                    "The operation is not supported using the adjoint "
                    "differentiation method");
        if ((ops_name[op_idx] == "QubitStateVector") ||
            (ops_name[op_idx] == "BasisState")) {
            continue; // ignore them
        }

        if (tp_it == tp_rend) {
            break; // All done
        }

        if (ops.hasParams(op_idx)) {
            if (current_param_idx == *tp_it) {
                // if current parameter is a trainable parameter
                mu_d.updateData(mu.getDataVector());
                const auto scalingFactor =
                    mu_d.applyGenerator(ops_name[op_idx],
                                        ops.getOpsWires()[op_idx],
                                        !ops.getOpsInverses()[op_idx]) *
                    (ops.getOpsInverses()[op_idx] ? -1 : 1);

                vec_out[trainable_param_idx] =
                    ComplexPrecisionT{0.0, scalingFactor} *
                    Util::innerProdC(mu_d.getDataVector(),
                                     lambda.getDataVector());
                --trainable_param_idx;
                ++tp_it;
            }
            --current_param_idx;
        }
        applyOperationAdj(lambda, ops, static_cast<size_t>(op_idx));
        applyOperationAdj(mu, ops, static_cast<size_t>(op_idx));
    }
};

/**
 * @brief
 */
template <typename PrecisionT>
auto fisherMatrix(const JacobianData<PrecisionT> &jd,
                  bool apply_operations = false) {
}

} // namespace Pennylane::Algorithms
