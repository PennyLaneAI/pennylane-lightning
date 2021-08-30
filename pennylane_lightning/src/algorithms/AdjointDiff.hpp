#pragma once

#include <complex>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "StateVector.hpp"
#include "StateVectorManaged.hpp"
#include "Util.hpp"

#include <iostream>

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
void applyGeneratorRX(SVType &sv, const std::vector<size_t> &wires) {
    sv.applyOperation("PauliX", wires, false);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorRY(SVType &sv, const std::vector<size_t> &wires) {
    sv.applyOperation("PauliY", wires, false);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorRZ(SVType &sv, const std::vector<size_t> &wires) {
    sv.applyOperation("PauliZ", wires, false);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorPhaseShift(SVType &sv, const std::vector<size_t> &wires) {
    sv.applyOperation(getP11<T>(), wires, false);
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorCRX(SVType &sv, const std::vector<size_t> &wires) {
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
void applyGeneratorCRY(SVType &sv, const std::vector<size_t> &wires) {
    const vector<size_t> internalIndices = sv.generateBitPatterns(wires);
    const vector<size_t> externalWires = sv.getIndicesAfterExclusion(wires);
    const vector<size_t> externalIndices =
        sv.generateBitPatterns(externalWires);
    for (const size_t &externalIndex : externalIndices) {
        std::complex<T> *shiftedState = sv.getData() + externalIndex;
        std::complex<T> v0 = shiftedState[internalIndices[0]];
        shiftedState[internalIndices[0]] = shiftedState[internalIndices[1]] = 0;
        shiftedState[internalIndices[2]] =
            -IMAG<T>() * shiftedState[internalIndices[3]];
        shiftedState[internalIndices[3]] = IMAG<T>() * v0;
    }
}

template <class T = double, class SVType = Pennylane::StateVector<T>>
void applyGeneratorCRZ(SVType &sv, const std::vector<size_t> &wires) {
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
void applyGeneratorControlledPhaseShift(SVType &sv,
                                        const std::vector<size_t> &wires) {
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

namespace Pennylane {
namespace Algorithms {

/**
 * @brief Utility struct for a observable operations used by AdjointJacobian
 * class.
 *
 */
template <class T = double> class ObsDatum {
  private:
    const std::vector<std::string> obs_name_;
    const std::vector<std::vector<T>> obs_params_;
    const std::vector<std::vector<size_t>> obs_wires_;

  public:
    /**
     * @brief Construct an ObsDatum object, representing a given observable.
     *
     * @param obs_name Name of each operation of the observable. Tensor product
     * observables have more than one operation.
     * @param obs_params Parameters for a given obserable opeartion ({} if
     * optional).
     * @param ops_wires Wires upon which to apply operation. Each observable
     * operation will eb a separate nested list.
     */
    ObsDatum(const std::vector<std::string> &obs_name,
             const std::vector<std::vector<T>> &obs_params,
             const std::vector<std::vector<size_t>> &obs_wires)
        : obs_name_{obs_name}, obs_params_{obs_params}, obs_wires_{
                                                            obs_wires} {};
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
    const std::vector<std::vector<T>> &getObsParams() const {
        return obs_params_;
    }
    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    const std::vector<std::vector<size_t>> &getObsWires() const {
        return obs_wires_;
    }
};

/**
 * @brief Utility class for encapsulating operations used by AdjointJacobian
 * class.
 *
 */
template <class T> struct OpsData {
    const std::vector<std::string> ops_name_;
    const std::vector<std::vector<T>> ops_params_;
    const std::vector<std::vector<size_t>> ops_wires_;
    const std::vector<bool> ops_inverses_;
    const std::vector<std::vector<std::complex<T>>> ops_matrices_;

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
          ops_inverses_{ops_inverses}, ops_matrices_{ops_matrices} {};

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
          ops_inverses_{ops_inverses}, ops_matrices_(ops_name.size()){};

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
};

/**
 * @brief Represent the logic for the adjoint Jacobian method of
 * arXiV:2009.02823
 *
 * @tparam T Floating-point precision.
 */
template <class T = double> class AdjointJacobian {
  private:
    typedef void (*GeneratorFunc)(
        StateVectorManaged<T> &sv,
        const std::vector<size_t> &wires); // function pointer type

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
     * @param sv1 Statevector <sv1|. Each datum will be conjugated.
     * @param sv2 Statevector |sv2>
     * @param jac Jacobian receiving the values.
     * @param num_elements Length of statevectors
     * @param scaling_coeff Generator coefficient for given gate derivative.
     * @param index Position of Jacobian to update.
     */
    inline void updateJacobian(const std::complex<T> *sv1,
                               const std::complex<T> *sv2, std::vector<T> &jac,
                               size_t num_elements, T scaling_coeff,
                               size_t index) {
        jac[index] =
            -2 * scaling_coeff * std::imag(innerProdC(sv1, sv2, num_elements));
    }
    /**
     * @brief Utility method to update the Jacobian at a given index by
     calculating the overlap between two given states.
     *
     * @see updateJacobian(const std::complex<T> *sv1,
                               const std::complex<T> *sv2, std::vector<T> &jac,
                               size_t num_elements, T scaling_coeff,
                               size_t index)
     */
    inline void updateJacobian(const std::vector<std::complex<T>> &sv1,
                               const std::vector<std::complex<T>> &sv2,
                               std::vector<T> &jac, size_t num_elements,
                               T scaling_coeff, size_t index) {
        PL_ASSERT(index < jac.size());
        jac[index] = -2 * scaling_coeff * std::imag(innerProdC(sv1, sv2));
    }

    /**
     * @brief Utility method to apply all operations from given `%OpsData<T>`
     * object to `%StateVectorManaged<T>`
     *
     * @param state Statevector to be updated.
     * @param operations Operations to apply.
     */
    inline void applyOperations(StateVectorManaged<T> &state,
                                const OpsData<T> &operations) {
        for (size_t op_idx = 0; op_idx < operations.getOpsName().size();
             op_idx++) {
            state.applyOperation(operations.getOpsName()[op_idx],
                                 operations.getOpsWires()[op_idx],
                                 operations.getOpsInverses()[op_idx],
                                 operations.getOpsParams()[op_idx]);
        }
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
        for (size_t j = 0; j < observable.getSize(); j++) {
            state.applyOperation(observable.getObsName()[j],
                                 observable.getObsWires()[j], false,
                                 observable.getObsParams()[j]);
        }
    }

  public:
    AdjointJacobian() {}

    /**
     * @brief Utility to create a given observable object.
     *
     * @param obs_name
     * @param obs_params
     * @param obs_wires
     * @return const ObsDatum<T>
     */
    const ObsDatum<T>
    createObs(const std::vector<std::string> &obs_name,
              const std::vector<std::vector<T>> &obs_params,
              const std::vector<std::vector<size_t>> &obs_wires) {
        return {obs_name, obs_params, obs_wires};
    }

    /**
     * @brief Utility to create a given operations object.
     *
     * @param ops_name
     * @param ops_params
     * @param ops_wires
     * @param ops_inverses
     * @param ops_matrices
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
     * @param psi
     * @param num_elements
     * @param jac
     * @param observables
     * @param operations
     * @param trainableParams
     * @param num_params
     */
    void adjointJacobian(const std::complex<T> *psi, size_t num_elements,
                         std::vector<T> &jac,
                         const std::vector<ObsDatum<T>> &observables,
                         const OpsData<T> &operations,
                         const vector<size_t> &trainableParams,
                         size_t num_params) {
        const size_t num_observables = observables.size();
        unsigned int trainableParamNumber = trainableParams.size() - 1;
        int current_param_idx = num_params - 1;

        // 1. Create $U_{1:p}\vert \lambda \rangle$
        StateVectorManaged<T> lambda(psi, num_elements);
        applyOperations(lambda, operations);

        // 2. Create observable-applied state-vectors
        std::vector<StateVectorManaged<T>> H_lambda(num_observables,
                                                    {lambda.getNumQubits()});

#pragma omp parallel for
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

                lambda.applyOperation(operations.getOpsName()[op_idx],
                                      operations.getOpsWires()[op_idx],
                                      !operations.getOpsInverses()[op_idx],
                                      operations.getOpsParams()[op_idx]);

                if (!operations.getOpsParams()[op_idx].empty()) {
                    if (std::find(trainableParams.begin(),
                                  trainableParams.end(),
                                  current_param_idx) != trainableParams.end()) {
                        // Apply generator function
                        generator_map.at(operations.getOpsName()[op_idx])(
                            mu, operations.getOpsWires()[op_idx]);
                        const T scalingFactor =
                            scaling_factors.at(operations.getOpsName()[op_idx]);

                        size_t index;
#pragma omp parallel for
                        for (size_t obs_idx = 0; obs_idx < num_observables;
                             obs_idx++) {
                            index = obs_idx * trainableParams.size() +
                                    trainableParamNumber;
                            updateJacobian(H_lambda[obs_idx].getData(),
                                           mu.getData(), jac, num_elements,
                                           scalingFactor, index);
                        }

                        trainableParamNumber--;
                    }
                    current_param_idx--;
                }

#pragma omp parallel for
                for (size_t obs_idx = 0; obs_idx < num_observables; obs_idx++) {
                    H_lambda[obs_idx].applyOperation(
                        operations.getOpsName()[op_idx],
                        operations.getOpsWires()[op_idx],
                        !operations.getOpsInverses()[op_idx],
                        operations.getOpsParams()[op_idx]);
                }
            }
        }
    }
};

} // namespace Algorithms
} // namespace Pennylane