// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
#include "AdjointJacobianBase.hpp"
#include "ObservablesKokkos.hpp"
#include <chrono>
#include <iostream>
#include <span>

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Algorithms;
using Pennylane::LightningKokkos::Util::getImagOfComplexInnerProduct;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Algorithms {
/**
 * @brief Kokkos-enabled adjoint Jacobian evaluator following the method of
 * arXiv:2009.02823
 *
 * @tparam StateVectorT State vector type.
 */
template <class StateVectorT>
class AdjointJacobian final
    : public AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>> {
  private:
    using BaseType =
        AdjointJacobianBase<StateVectorT, AdjointJacobian<StateVectorT>>;
    using typename BaseType::ComplexT;
    using typename BaseType::PrecisionT;

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param idx Linear Jacobian index.
     */
    inline void updateJacobian(StateVectorT &sv1, StateVectorT &sv2,
                               std::span<PrecisionT> &jac,
                               PrecisionT scaling_coeff, std::size_t idx) {
        auto element = -2 * scaling_coeff *
                       getImagOfComplexInnerProduct<PrecisionT>(sv1.getView(),
                                                                sv2.getView());
        jac[idx] = element;
    }

  public:
    AdjointJacobian() = default;

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies, one per required observable. The `operations`
     * will be applied to the internal statevector copies, with the operation
     * indices participating in the gradient calculations given in
     * `trainableParams`, and the overall number of parameters for the gradient
     * calculation provided within `num_params`. The resulting row-major ordered
     * `jac` matrix representation will be of size `jd.getSizeStateVec() *
     * jd.getObservables().size()`. OpenMP is used to enable independent
     * operations to be offloaded to threads.
     *
     * @note Only gates with pre-defined generators can be differentiated.
     * For example, `QubitUnitary` is not differentiable as there is no
     * generator defined for this gate.
     *
     * @param jac Preallocated vector for Jacobian data results.
     * @param jd JacobianData represents the QuantumTape to differentiate.
     * @param apply_operations Indicate whether to apply operations to tape.psi
     * prior to calculation.
     */
    void adjointJacobian(std::span<PrecisionT> jac,
                         const JacobianData<StateVectorT> &jd,
                         const StateVectorT &ref_data,
                         bool apply_operations = false) {

        std::chrono::duration<double, std::milli> zero_time = std::chrono::duration<double, std::milli>::zero();

        auto total_start = std::chrono::steady_clock::now();
        auto total_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> total_time = zero_time;

        auto step_start = std::chrono::steady_clock::now();
        auto step_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> step_time = zero_time;

        {
            std::cout << "adjointJacobian::start" << std::endl;
        }

        total_start = std::chrono::steady_clock::now();
        step_start = std::chrono::steady_clock::now();

        const OpsData<StateVectorT> &ops = jd.getOperations();
        const std::vector<std::string> &ops_name = ops.getOpsName();

        const auto &obs = jd.getObservables();
        const std::size_t num_observables = obs.size();

        // We can assume the trainable params are sorted (from Python)
        const std::vector<std::size_t> &tp = jd.getTrainableParams();
        const std::size_t tp_size = tp.size();
        const std::size_t num_param_ops = ops.getNumParOps();

        if (!jd.hasTrainableParams()) {
            return;
        }

        PL_ABORT_IF_NOT(
            jac.size() == tp_size * num_observables,
            "The size of preallocated jacobian must be same as "
            "the number of trainable parameters times the number of "
            "observables provided.");

        // Track positions within par and non-par operations
        std::size_t trainableParamNumber = tp_size - 1;
        std::size_t current_param_idx =
            num_param_ops - 1; // total number of parametric ops
        auto tp_it = tp.rbegin();
        const auto tp_rend = tp.rend();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorT lambda{ref_data};

        // Apply given operations to statevector if requested
        if (apply_operations) {
            std::cout << "adjointJacobian::apply_operations" << std::endl;
            BaseType::applyOperations(lambda, ops);
        }

        StateVectorT mu{lambda.getNumQubits()};

        step_end = std::chrono::steady_clock::now();
        step_time = step_end - step_start;
        {
            std::cout << "adjointJacobian::allocate (ms) " << step_time.count() << std::endl;
        }
        step_start = std::chrono::steady_clock::now();

        // Create observable-applied state-vectors
        std::vector<StateVectorT> H_lambda(num_observables,
                                           StateVectorT(lambda.getNumQubits()));
        BaseType::applyObservables(H_lambda, lambda, obs);

        step_end = std::chrono::steady_clock::now();
        step_time = step_end - step_start;
        {
            std::cout << "adjointJacobian::applyObs (ms) " << step_time.count() << std::endl;
        }
        step_start = std::chrono::steady_clock::now();

        int adj1_cnt = 0;
        auto adj1_start = std::chrono::steady_clock::now();
        auto adj1_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> adj1_time = zero_time;

        int adj2_cnt = 0;
        auto adj2_start = std::chrono::steady_clock::now();
        auto adj2_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> adj2_time = zero_time;

        int updt_cnt = 0;
        auto updt_start = std::chrono::steady_clock::now();
        auto updt_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> updt_time = zero_time;

        int deriv_cnt = 0;
        auto deriv_start = std::chrono::steady_clock::now();
        auto deriv_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> deriv_time = zero_time;

        int jacob_cnt = 0;
        auto jacob_start = std::chrono::steady_clock::now();
        auto jacob_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> jacob_time = zero_time;

        int trnsp_cnt = 0;
        auto trnsp_start = std::chrono::steady_clock::now();
        auto trnsp_end = std::chrono::steady_clock::now();
        std::chrono::duration<double, std::milli> trnsp_time = zero_time;

        for (int op_idx = static_cast<int>(ops_name.size() - 1); op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(ops.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((ops_name[op_idx] == "StatePrep") ||
                (ops_name[op_idx] == "BasisState")) {
                continue;
            }
            if (tp_it == tp_rend) {
                break; // All done
            }
            if (ops.hasParams(op_idx)) {
                if (current_param_idx == *tp_it) {

                    updt_start = std::chrono::steady_clock::now();
                    mu.updateData(lambda);
                    updt_end = std::chrono::steady_clock::now();
                    updt_time += updt_end - updt_start;
                    updt_cnt++;

                    deriv_start = std::chrono::steady_clock::now();
                    const PrecisionT scalingFactor =
                        (ops.getOpsControlledWires()[op_idx].empty())
                            ? BaseType::applyGenerator(
                                  mu, ops.getOpsName()[op_idx],
                                  ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1)
                            : BaseType::applyGenerator(
                                  mu, ops.getOpsName()[op_idx],
                                  ops.getOpsControlledWires()[op_idx],
                                  ops.getOpsControlledValues()[op_idx],
                                  ops.getOpsWires()[op_idx],
                                  !ops.getOpsInverses()[op_idx]) *
                                  (ops.getOpsInverses()[op_idx] ? -1 : 1);
                    deriv_end = std::chrono::steady_clock::now();
                    deriv_time += deriv_end - deriv_start;
                    deriv_cnt++;

                    jacob_start = std::chrono::steady_clock::now();
                    for (std::size_t obs_idx = 0; obs_idx < num_observables;
                         obs_idx++) {
                        const std::size_t idx =
                            trainableParamNumber + obs_idx * tp_size;
                        updateJacobian(H_lambda[obs_idx], mu, jac,
                                       scalingFactor, idx);
                    }
                    jacob_end = std::chrono::steady_clock::now();
                    jacob_time += jacob_end - jacob_start;
                    jacob_cnt++;

                    trainableParamNumber--;
                    ++tp_it;
                }
                current_param_idx--;
            }

            adj1_start = std::chrono::steady_clock::now();
            BaseType::applyOperationAdj(lambda, ops, op_idx);
            adj1_end = std::chrono::steady_clock::now();
            adj1_time += adj1_end - adj1_start;
            adj1_cnt++;

            adj2_start = std::chrono::steady_clock::now();
            this->applyOperationsAdj(H_lambda, ops,
                                     static_cast<std::size_t>(op_idx));
            adj2_end = std::chrono::steady_clock::now();
            adj2_time += adj2_end - adj2_start;
            adj2_cnt++;
        }

        step_end = std::chrono::steady_clock::now();
        step_time = step_end - step_start;

        total_end = std::chrono::steady_clock::now();
        total_time = total_end - total_start;

        {
            std::cout << "adjointJacobian::loop (ms) " << step_time.count() << std::endl;
            std::cout << "adjointJacobian::_mu.updt (ms) " << updt_time.count() << " cnt=" << updt_cnt << std::endl;
            std::cout << "adjointJacobian::_deriv (ms) " << deriv_time.count() << " cnt=" << deriv_cnt << std::endl;
            std::cout << "adjointJacobian::_jacob (ms) " << jacob_time.count() << " cnt=" << jacob_cnt << std::endl;
            std::cout << "adjointJacobian::_adj_1 (ms) " << adj1_time.count() << " cnt=" << adj1_cnt << std::endl;
            std::cout << "adjointJacobian::_adj_2 (ms) " << adj2_time.count() << " cnt=" << adj2_cnt << std::endl;
            std::cout << "adjointJacobian::_trnsp (ms) " << trnsp_time.count() << " cnt=" << trnsp_cnt << std::endl;
            std::cout << "adjointJacobian::total (ms) " << total_time.count() << std::endl;
            std::cout << "adjointJacobian::end" << std::endl;
        }
    }
};

} // namespace Pennylane::LightningKokkos::Algorithms
