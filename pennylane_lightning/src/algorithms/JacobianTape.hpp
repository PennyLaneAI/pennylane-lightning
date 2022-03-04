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
#include <utility>
#include <variant>
#include <vector>

namespace Pennylane::Algorithms {

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
     * @param obs_params Parameters for a given observable operation ({} if
     * optional).
     * @param obs_wires Wires upon which to apply operation. Each observable
     * operation will be a separate nested list.
     */
    ObsDatum(std::vector<std::string> obs_name,
             std::vector<param_var_t> obs_params,
             std::vector<std::vector<size_t>> obs_wires)
        : obs_name_{std::move(obs_name)},
          obs_params_(std::move(obs_params)), obs_wires_{
                                                  std::move(obs_wires)} {};

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_name_.size(); }
    /**
     * @brief Get the name of the observable operations.
     *
     * @return const std::vector<std::string>&
     */
    [[nodiscard]] auto getObsName() const -> const std::vector<std::string> & {
        return obs_name_;
    }
    /**
     * @brief Get the parameters for the observable operations.
     *
     * @return const std::vector<std::vector<T>>&
     */
    [[nodiscard]] auto getObsParams() const
        -> const std::vector<param_var_t> & {
        return obs_params_;
    }
    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getObsWires() const
        -> const std::vector<std::vector<size_t>> & {
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
    OpsData(std::vector<std::string> ops_name,
            const std::vector<std::vector<T>> &ops_params,
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses,
            std::vector<std::vector<std::complex<T>>> ops_matrices)
        : ops_name_{std::move(ops_name)}, ops_params_{ops_params},
          ops_wires_{std::move(ops_wires)},
          ops_inverses_{std::move(ops_inverses)}, ops_matrices_{
                                                      std::move(ops_matrices)} {
        num_par_ops_ = 0;
        for (const auto &p : ops_params) {
            if (!p.empty()) {
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
            std::vector<std::vector<size_t>> ops_wires,
            std::vector<bool> ops_inverses)
        : ops_name_{ops_name}, ops_params_{ops_params},
          ops_wires_{std::move(ops_wires)}, ops_inverses_{std::move(
                                                ops_inverses)},
          ops_matrices_(ops_name.size()) {
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
    [[nodiscard]] auto getSize() const -> size_t { return ops_name_.size(); }

    /**
     * @brief Get the names of the operations to be applied.
     *
     * @return const std::vector<std::string>&
     */
    [[nodiscard]] auto getOpsName() const -> const std::vector<std::string> & {
        return ops_name_;
    }
    /**
     * @brief Get the (optional) parameters for each operation. Given entries
     * are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<T>>&
     */
    [[nodiscard]] auto getOpsParams() const
        -> const std::vector<std::vector<T>> & {
        return ops_params_;
    }
    /**
     * @brief Get the wires for each operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getOpsWires() const
        -> const std::vector<std::vector<size_t>> & {
        return ops_wires_;
    }
    /**
     * @brief Get the adjoint flag for each operation.
     *
     * @return const std::vector<bool>&
     */
    [[nodiscard]] auto getOpsInverses() const -> const std::vector<bool> & {
        return ops_inverses_;
    }
    /**
     * @brief Get the numerical matrix for a given unsupported operation. Given
     * entries are empty ({}) if not required.
     *
     * @return const std::vector<std::vector<std::complex<T>>>&
     */
    [[nodiscard]] auto getOpsMatrices() const
        -> const std::vector<std::vector<std::complex<T>>> & {
        return ops_matrices_;
    }

    /**
     * @brief Notify if the operation at a given index is parametric.
     *
     * @param index Operation index.
     * @return true Gate is parametric (has parameters).
     * @return false Gate in non-parametric.
     */
    [[nodiscard]] inline auto hasParams(size_t index) const -> bool {
        return !ops_params_[index].empty();
    }

    /**
     * @brief Get the number of parametric operations.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumParOps() const -> size_t { return num_par_ops_; }

    /**
     * @brief Get the number of non-parametric ops.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumNonParOps() const -> size_t {
        return num_nonpar_ops_;
    }
};

/**
 * @brief Represent the serialized data of a QuantumTape to differentiate
 *
 * @param num_parameters Number of parameters in the Tape.
 * @param num_elements Length of the statevector data.
 * @param psi Pointer to the statevector data.
 * @param observables Observables for which to calculate Jacobian.
 * @param operations Operations used to create given state.
 * @param trainableParams List of parameters participating in Jacobian
 * calculation.
 */
template <class T> class JacobianData {
  private:
    size_t num_parameters;
    size_t num_elements;
    const std::complex<T> *psi;
    const std::vector<ObsDatum<T>> observables;
    const OpsData<T> operations;
    const std::vector<size_t> trainableParams;

  public:
    /**
     * @brief Construct a JacobianData object
     *
     * @param num_params Number of parameters in the Tape.
     * @param num_elem Length of the statevector data.
     * @param ps Pointer to the statevector data.
     * @param obs Observables for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainP List of parameters participating in Jacobian
     * calculation.
     */
    JacobianData(size_t num_params, size_t num_elem, std::complex<T> *ps,
                 std::vector<ObsDatum<T>> obs, OpsData<T> ops,
                 std::vector<size_t> trainP)
        : num_parameters(num_params), num_elements(num_elem), psi(ps),
          observables(std::move(obs)), operations(std::move(ops)),
          trainableParams(std::move(trainP)) {}

    /**
     * @brief Get Number of parameters in the Tape.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumParams() const -> size_t { return num_parameters; }

    /**
     * @brief Get the length of the statevector data.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSizeStateVec() const -> size_t {
        return num_elements;
    }

    /**
     * @brief Get the pointer to the statevector data.
     *
     * @return std::complex<T> *
     */
    [[nodiscard]] auto getPtrStateVec() const -> const std::complex<T> * {
        return psi;
    }

    /**
     * @brief Get observables for which to calculate Jacobian.
     *
     * @return std::vector<ObsDatum<T>>&
     */
    [[nodiscard]] auto getObservables() const
        -> const std::vector<ObsDatum<T>> & {
        return observables;
    }

    /**
     * @brief Get the number of observables for which to calculate
     * Jacobian.
     *
     * @return size_t
     */
    [[nodiscard]] auto getNumObservables() const -> size_t {
        return observables.size();
    }

    /**
     * @brief Get operations used to create given state.
     *
     * @return OpsData<T>&
     */
    [[nodiscard]] auto getOperations() const -> const OpsData<T> & {
        return operations;
    }

    /**
     * @brief Get list of parameters participating in Jacobian
     * calculation.
     *
     * @return std::vector<size_t>&
     */
    [[nodiscard]] auto getTrainableParams() const
        -> const std::vector<size_t> & {
        return trainableParams;
    }

    /**
     * @brief Get if the number of parameters participating in Jacobian
     * calculation is zero.
     *
     * @return true If it has trainable parameters; false otherwise.
     */
    [[nodiscard]] auto hasTrainableParams() const -> bool {
        return !trainableParams.empty();
    }
};
} // namespace Pennylane::Algorithms
