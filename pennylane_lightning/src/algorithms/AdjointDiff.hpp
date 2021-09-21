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

#include <complex>
#include <cstring>
#include <numeric>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Error.hpp"
#include "StateVector.hpp"
#include "StateVectorManaged.hpp"
#include "Util.hpp"

#include <iostream>

/// @cond DEV
namespace {

using namespace Pennylane;
using namespace Pennylane::Util;

template <class T> static constexpr std::vector<std::complex<T>> getP00() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getP11() {
    return {ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorRX(SVType &sv, const std::vector<size_t> &wires,
                      const bool adj = false) {
    sv.applyOperation("PauliX", wires, adj);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorRY(SVType &sv, const std::vector<size_t> &wires,
                      const bool adj = false) {
    sv.applyOperation("PauliY", wires, adj);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorRZ(SVType &sv, const std::vector<size_t> &wires,
                      const bool adj = false) {
    sv.applyOperation("PauliZ", wires, adj);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorPhaseShift(SVType &sv, const std::vector<size_t> &wires,
                              const bool adj = false) {
    sv.applyOperation(getP11<T>(), wires, adj);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorCRX(SVType &sv, const std::vector<size_t> &wires,
                       [[maybe_unused]] const bool adj = false) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        std::swap(shiftedState[internalIndices[2]],
                  shiftedState[internalIndices[3]]);
    }
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorCRY(SVType &sv, const std::vector<size_t> &wires,
                       [[maybe_unused]] const bool adj = false) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        std::complex<T> v0 = shiftedState[internalIndices[2]];
        shiftedState[internalIndices[0]] = ZERO<T>();
        shiftedState[internalIndices[1]] = ZERO<T>();
        shiftedState[internalIndices[2]] =
            -IMAG<T>() * shiftedState[internalIndices[3]];
        shiftedState[internalIndices[3]] = IMAG<T>() * v0;
    }
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorCRZ(SVType &sv, const std::vector<size_t> &wires,
                       [[maybe_unused]] const bool adj = false) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[3]] *= -1;
    }
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorControlledPhaseShift(
    SVType &sv, const std::vector<size_t> &wires,
    [[maybe_unused]] const bool adj = false) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        shiftedState[internalIndices[0]] = 0;
        shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[2]] = 0;
    }
}

} // namespace
/// @endcond

namespace Pennylane {
namespace Algorithms {

/**
 * @brief Utility struct for observable operations used by AdjointJacobian
 * class.
 *
 */
template <class T = double> class ObsDatum {
  public:
    /**
     * @brief Variant type of stored parameter data.
     */
    using param_var_t = std::variant<std::monostate, std::vector<T>,
                                     std::vector<std::complex<T>>>;

    /**
     * @brief Copy constructor for an ObsDatum object, representing a given
     * observable.
     *
     * @param obs_name Name of each operation of the observable. Tensor product
     * observables have more than one operation.
     * @param obs_params Parameters for a given obserable operation ({} if
     * optional).
     * @param obs_wires Wires upon which to apply operation. Each observable
     * operation will be a separate nested list.
     */
    ObsDatum(const std::vector<std::string> &obs_name,
             const std::vector<param_var_t> &obs_params,
             const std::vector<std::vector<size_t>> &obs_wires)
        : obs_name_{obs_name},
          obs_params_(obs_params), obs_wires_{obs_wires} {};

    /**
     * @brief Move constructor for an ObsDatum object, representing a given
     * observable.
     *
     * @param obs_name Name of each operation of the observable. Tensor product
     * observables have more than one operation.
     * @param obs_params Parameters for a given obserable operation ({} if
     * optional).
     * @param obs_wires Wires upon which to apply operation. Each observable
     * operation will be a separate nested list.
     */
    ObsDatum(std::vector<std::string> &&obs_name,
             std::vector<param_var_t> &&obs_params,
             std::vector<std::vector<size_t>> &&obs_wires)
        : obs_name_{std::move(obs_name)}, obs_params_{std::move(obs_params)},
          obs_wires_{std::move(obs_wires)} {};

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    size_t getSize() const { return obs_name_.size(); }
    /**
     * @brief Get the name of the observable operations.
     *
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string> &getObsName() const { return obs_name_; }
    /**
     * @brief Get the parameters for the observable operations.
     *
     * @return const std::vector<std::vector<T>>&
     */
    const std::vector<param_var_t> &getObsParams() const { return obs_params_; }
    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    const std::vector<std::vector<size_t>> &getObsWires() const {
        return obs_wires_;
    }

  private:
    const std::vector<std::string> obs_name_;
    const std::vector<param_var_t> obs_params_;
    const std::vector<std::vector<size_t>> obs_wires_;
};

/**
 * @brief Utility class for encapsulating operations used by AdjointJacobian
 * class.
 *
 */
template <class T> class OpsData {
  private:
    size_t num_par_ops_;
    size_t num_nonpar_ops_;
    const std::vector<std::string> ops_name_;
    const std::vector<std::vector<T>> ops_params_;
    const std::vector<std::vector<size_t>> ops_wires_;
    const std::vector<bool> ops_inverses_;
    const std::vector<std::vector<std::complex<T>>> ops_matrices_;

  public:
    /**
     * @brief Construct an OpsData object, representing the serialized
     * operations to apply upon the `%StateVector`.
     *
     * @param ops_name Name of each operation to apply.
     * @param ops_params Parameters for a given operation ({} if optional).
     * @param ops_wires Wires upon which to apply operation
     * @param ops_inverses Value to represent whether given operation is
     * adjoint.
     * @param ops_matrices Numerical representation of given matrix if not
     * supported.
     */
    OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<T>> &ops_params,
            const std::vector<std::vector<size_t>> &ops_wires,
            const std::vector<bool> &ops_inverses,
            const std::vector<std::vector<std::complex<T>>> &ops_matrices)
        : ops_name_{ops_name}, ops_params_{ops_params}, ops_wires_{ops_wires},
          ops_inverses_{ops_inverses}, ops_matrices_{ops_matrices} {
        num_par_ops_ = 0;
        for (const auto &p : ops_params) {
            if (p.size() > 0) {
                num_par_ops_++;
            }
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Construct an OpsData object, representing the serialized
     operations to apply upon the `%StateVector`.
     *
     * @see  OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<T>> &ops_params,
            const std::vector<std::vector<size_t>> &ops_wires,
            const std::vector<bool> &ops_inverses,
            const std::vector<std::vector<std::complex<T>>> &ops_matrices)
     */
    OpsData(const std::vector<std::string> &ops_name,
            const std::vector<std::vector<T>> &ops_params,
            const std::vector<std::vector<size_t>> &ops_wires,
            const std::vector<bool> &ops_inverses)
        : ops_name_{ops_name}, ops_params_{ops_params}, ops_wires_{ops_wires},
          ops_inverses_{ops_inverses}, ops_matrices_(ops_name.size()) {
        num_par_ops_ = 0;
        for (const auto &p : ops_params) {
            if (p.size() > 0) {
                num_par_ops_++;
            }
        }
        num_nonpar_ops_ = ops_params.size() - num_par_ops_;
    };

    /**
     * @brief Get the number of operations to be applied.
     *
     * @return size_t Number of operations.
     */
    size_t getSize() const { return ops_name_.size(); }

    /**
     * @brief Get the names of the operations to be applied.
     *
     * @return const std::vector<std::string>&
     */
    const std::vector<std::string> &getOpsName() const { return ops_name_; }
    /**
     * @brief Get the (optional) parameters for each operation. Given entries
     * are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<T>>&
     */
    const std::vector<std::vector<T>> &getOpsParams() const {
        return ops_params_;
    }
    /**
     * @brief Get the wires for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    const std::vector<std::vector<size_t>> &getOpsWires() const {
        return ops_wires_;
    }
    /**
     * @brief Get the adjoint flag for each operation.
     *
     * @return const std::vector<bool>&
     */
    const std::vector<bool> &getOpsInverses() const { return ops_inverses_; }
    /**
     * @brief Get the numerical matrix for a given unsupported operation. Given
     * entries are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<std::complex<T>>>&
     */
    const std::vector<std::vector<std::complex<T>>> &getOpsMatrices() const {
        return ops_matrices_;
    }

    /**
     * @brief Notify if the operation at a given index is parametric.
     *
     * @param index Operation index.
     * @return true Gate is parametric (has parameters).
     * @return false Gate in non-parametric.
     */
    inline bool hasParams(size_t index) const {
        return !ops_params_[index].empty();
    }

    /**
     * @brief Get the number of parametric operations.
     *
     * @return size_t
     */
    size_t getNumParOps() const { return num_par_ops_; }

    /**
     * @brief Get the number of non-parametric ops.
     *
     * @return size_t
     */
    size_t getNumNonParOps() const { return num_nonpar_ops_; }
};

/**
 * @brief Represent the logic for the adjoint Jacobian method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobian {
  private:
    typedef void (*GeneratorFunc)(StateVectorManaged<T> &sv,
                                  const std::vector<size_t> &wires,
                                  const bool adj); // function pointer type

    // Holds the mapping from gate labels to associated generator functions.
    const std::unordered_map<std::string, GeneratorFunc> generator_map{
        {"RX", &::applyGeneratorRX<T, StateVectorManaged<T>>},
        {"RY", &::applyGeneratorRY<T, StateVectorManaged<T>>},
        {"RZ", &::applyGeneratorRZ<T, StateVectorManaged<T>>},
        {"PhaseShift", &::applyGeneratorPhaseShift<T, StateVectorManaged<T>>},
        {"CRX", &::applyGeneratorCRX<T, StateVectorManaged<T>>},
        {"CRY", &::applyGeneratorCRY<T, StateVectorManaged<T>>},
        {"CRZ", &::applyGeneratorCRZ<T, StateVectorManaged<T>>},
        {"ControlledPhaseShift",
         &::applyGeneratorControlledPhaseShift<T, StateVectorManaged<T>>}};

    // Holds the mappings from gate labels to associated generator coefficients.
    const std::unordered_map<std::string, T> scaling_factors{
        {"RX", -0.5},  {"RY", -0.5},
        {"RZ", -0.5},  {"PhaseShift", 1},
        {"CRX", -0.5}, {"CRY", -0.5},
        {"CRZ", -0.5}, {"ControlledPhaseShift", 1}};

    /**
     * @brief Utility method to update the Jacobian at a given index by
     * calculating the overlap between two given states.
     *
     * @param sv1 Statevector <sv1|. Data will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param obs_index Observable index position of Jacobian to update.
     * @param param_index Parameter index position of Jacobian to update.
     */
    inline void updateJacobian(const StateVectorManaged<T> &sv1,
                               const StateVectorManaged<T> &sv2,
                               std::vector<std::vector<T>> &jac,
                               T scaling_coeff, size_t obs_index,
                               size_t param_index) {
        jac[obs_index][param_index] =
            -2 * scaling_coeff *
            std::imag(innerProdC(sv1.getDataVector(), sv2.getDataVector()));
    }

    /**
     * @brief Utility method to apply all operations from given `%OpsData<T>`
     * object to `%StateVectorManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param adj Take the adjoint of the given operations.
     */
    inline void applyOperations(StateVectorManaged<T> &state,
                                const OpsData<T> &operations,
                                bool adj = false) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx] ^ adj,
                                 operations.getOpsParams()[op_idx]);
        }
    }
    /**
     * @brief Utility method to apply the adjoint indexed operation from
     * `%OpsData<T>` object to `%StateVectorManaged<T>`.
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     * @param op_idx Adjointed operation index to apply.
     */
    inline void applyOperationAdj(StateVectorManaged<T> &state,
                                  const OpsData<T> &operations, size_t op_idx) {
        state.applyOperation(operations.getOpsName()[op_idx],
                             operations.getOpsWires()[op_idx],
                             !operations.getOpsInverses()[op_idx],
                             operations.getOpsParams()[op_idx]);
    }

    /**
     * @brief Utility method to apply a given operations from given
     * `%ObsDatum<T>` object to `%StateVectorManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param observable Observable to apply.
     */
    inline void applyObservable(StateVectorManaged<T> &state,
                                const ObsDatum<T> &observable) {
        using namespace Pennylane::Util;
        for (size_t j = 0; j < observable.getSize(); j++) {
            if (!observable.getObsParams().empty()) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        // Apply supported gate with given params
                        if constexpr (std::is_same_v<p_t, std::vector<T>>) {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false, param);
                        }
                        // Apply provided matrix
                        else if constexpr (std::is_same_v<
                                               p_t,
                                               std::vector<std::complex<T>>>) {
                            state.applyOperation(
                                param, observable.getObsWires()[j], false);
                        } else {
                            state.applyOperation(observable.getObsName()[j],
                                                 observable.getObsWires()[j],
                                                 false);
                        }
                    },
                    observable.getObsParams()[j]);
            } else { // Offloat to SV dispatcher if no parameters provided
                state.applyOperation(observable.getObsName()[j],
                                     observable.getObsWires()[j], false);
            }
        }
    }

    /**
     * @brief Inline utility to assist with getting the Jacobian index offset.
     *
     * @param obs_index
     * @param tp_index
     * @param tp_size
     * @return size_t
     */
    inline size_t getJacIndex(size_t obs_index, size_t tp_index,
                              size_t tp_size) {
        return obs_index * tp_size + tp_index;
    }

    /**
     * @brief Copies complex data array into a `%vector` of the same dimension.
     *
     * @param input_state
     * @param state_length
     * @return std::vector<std::complex<T>>
     */
    std::vector<std::complex<T>>
    copyStateData(const std::complex<T> *input_state, size_t state_length) {
        return {input_state, input_state + state_length};
    }

    /**
     * @brief Applies the gate generator for a given parameteric gate. Returns
     * the associated scaling coefficient.
     *
     * @param sv Statevector data to operate upon.
     * @param op_name Name of parametric gate.
     * @param wires Wires to operate upon.
     * @param adj Indicate whether to take the adjoint of the operation.
     * @return T Generator scaling coefficient.
     */
    inline T applyGenerator(StateVectorManaged<T> &sv,
                            const std::string &op_name,
                            const std::vector<size_t> &wires, const bool adj) {
        generator_map.at(op_name)(sv, wires, adj);
        return scaling_factors.at(op_name);
    }

  public:
    AdjointJacobian() {}

    /**
     * @brief Utility to create a given operations object.
     *
     * @param ops_name Name of operations.
     * @param ops_params Parameters for each operation in ops_name.
     * @param ops_wires Wires for each operation in ops_name.
     * @param ops_inverses Indicate whether to take adjoint of each operation in
     * ops_name.
     * @param ops_matrices Matrix definition of an operation if unsupported.
     * @return const OpsData<T>
     */
    const OpsData<T> createOpsData(
        const std::vector<std::string> &ops_name,
        const std::vector<std::vector<T>> &ops_params,
        const std::vector<std::vector<size_t>> &ops_wires,
        const std::vector<bool> &ops_inverses,
        const std::vector<std::vector<std::complex<T>>> &ops_matrices = {{}}) {
        return {ops_name, ops_params, ops_wires, ops_inverses, ops_matrices};
    }

    /**
     * @brief Calculates the Jacobian for the statevector for the selected set
     * of parametric gates.
     *
     * For the statevector data associated with `psi` of length `num_elements`,
     * we make internal copies to a `%StateVectorManaged<T>` object, with one
     * per required observable. The `operations` will be applied to the internal
     * statevector copies, with the operation indices participating in the
     * gradient calculations given in `trainableParams`, and the overall number
     * of parameters for the gradient calculation provided within `num_params`.
     * The resulting row-major ordered `jac` matrix representation will be of
     * size `trainableParams.size() * observables.size()`. OpenMP is used to
     * enable independent operations to be offloaded to threads.
     *
     * @param psi Pointer to the statevector data.
     * @param num_elements Length of the statevector data.
     * @param jac Preallocated vector for Jacobian data results.
     * @param observables Observables for which to calculate Jacobian.
     * @param operations Operations used to create given state.
     * @param trainableParams List of parameters participating in Jacobian
     * calculation.
     * @param apply_operations Indicate whether to apply operations to psi prior
     * to calculation.
     */
    void adjointJacobian(const std::complex<T> *psi, size_t num_elements,
                         std::vector<std::vector<T>> &jac,
                         const std::vector<ObsDatum<T>> &observables,
                         const OpsData<T> &operations,
                         const std::set<size_t> &trainableParams,
                         bool apply_operations = false) {
        PL_ABORT_IF(trainableParams.empty(),
                    "No trainable parameters provided.");

        // Track positions within par and non-par operations
        size_t num_observables = observables.size();
        size_t trainableParamNumber = trainableParams.size() - 1;
        size_t current_param_idx =
            operations.getNumParOps() - 1; // total number of parametric ops

        auto tp_it = trainableParams.end();

        // Create $U_{1:p}\vert \lambda \rangle$
        StateVectorManaged<T> lambda(psi, num_elements);

        // Apply given operations to statevector if requested
        if (apply_operations)
            applyOperations(lambda, operations);

        // Create observable-applied state-vectors
        std::vector<StateVectorManaged<T>> H_lambda(num_observables,
                                                    {lambda.getNumQubits()});
#if defined _OPENMP
#pragma omp parallel for
#endif
        for (size_t h_i = 0; h_i < num_observables; h_i++) {
            H_lambda[h_i].updateData(lambda.getDataVector());
            applyObservable(H_lambda[h_i], observables[h_i]);
        }

        StateVectorManaged<T> mu(lambda.getNumQubits());

        for (int op_idx = operations.getOpsName().size() - 1; op_idx >= 0;
             op_idx--) {
            PL_ABORT_IF(operations.getOpsParams()[op_idx].size() > 1,
                        "The operation is not supported using the adjoint "
                        "differentiation method");
            if ((operations.getOpsName()[op_idx] != "QubitStateVector") &&
                (operations.getOpsName()[op_idx] != "BasisState")) {
                mu.updateData(lambda.getDataVector());
                applyOperationAdj(lambda, operations, op_idx);

                if (operations.hasParams(op_idx)) {
                    if (std::find(trainableParams.begin(), tp_it,
                                  current_param_idx) != tp_it) {
                        const T scalingFactor =
                            applyGenerator(
                                mu, operations.getOpsName()[op_idx],
                                operations.getOpsWires()[op_idx],
                                !operations.getOpsInverses()[op_idx]) *
                            (2 * (0b1 ^ operations.getOpsInverses()[op_idx]) -
                             1);
#if defined _OPENMP
#pragma omp parallel for
#endif
                        for (size_t obs_idx = 0; obs_idx < num_observables;
                             obs_idx++) {
                            updateJacobian(H_lambda[obs_idx], mu, jac,
                                           scalingFactor, obs_idx,
                                           trainableParamNumber);
                        }
                        trainableParamNumber--;
                        std::advance(tp_it, -1);
                    }
                    current_param_idx--;
                }
#if defined _OPENMP
#pragma omp parallel for
#endif
                for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                    applyOperationAdj(H_lambda[obs_idx], operations, op_idx);
                }
            }
        }
    }
};

} // namespace Algorithms
} // namespace Pennylane