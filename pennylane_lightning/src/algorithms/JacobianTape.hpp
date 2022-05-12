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

#include "Macros.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <complex>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Pennylane::Algorithms {

template <typename T> class Observable {
  protected:
    Observable() = default;
    Observable(Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorManaged<T> &sv) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;
};

/**
 * @brief Utility class for observable operations used by the adjoint
 * differentiation method.
 *
 * @tparam T Floating point type
 */
template <typename T> class NamedObs final : public Observable<T> {
  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

  public:
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param arg1 Name of the observable.
     * @param arg2 Wires upon which to apply operation.
     */
    template <typename T1, typename T2>
    NamedObs(T1 &&arg1, T2 &&arg2)
        : obs_name_{std::forward<T1>(arg1)}, wires_{std::forward<T2>(arg2)} {}

    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param arg1 Name of the observable.
     * @param arg2 Argument to construct wires.
     * @param arg3 Argument to construct parameters
     */
    template <typename T1, typename T2, typename T3>
    NamedObs(T1 &&arg1, T2 &&arg2, T3 &&arg3)
        : obs_name_{std::forward<T1>(arg1)}, wires_{std::forward<T2>(arg2)},
          params_{std::forward<T3>(arg3)} {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        return obs_name_;
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> final {
        return wires_;
    }

    void applyInPlace(StateVectorManaged<T> &sv) const final {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Utility class for observable operations used by the adjoint
 * differentiation method.
 *
 * @tparam T Floating point type
 */
template <typename T> class HermitianObs final : public Observable<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    MatrixT matrix_;
    std::vector<size_t> wires_;

  public:
    /**
     * @brief Create Hermitian observable
     *
     * @param arg1 Matrix in row major format.
     * @param arg2 Wires the observable applies to.
     */
    template <typename T1, typename T2>
    HermitianObs(T1 &&arg1, T2 &&arg2)
        : matrix_{std::forward<T1>(arg1)}, wires_{std::forward<T2>(arg2)} {}

    [[nodiscard]] auto getMatrix() const -> const MatrixT & { return matrix_; }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> final {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string final {
        return "Hermitian";
    }

    void applyInPlace(StateVectorManaged<T> &sv) const final {
        sv.applyMatrix(matrix_, wires_);
    }
};

template <typename T> class TensorProdObs final : public Observable<T> {
  private:
    std::vector<std::shared_ptr<Observable<T>>> obs_;

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {}

    /**
     * @brief Get the number of operations in observable.
     *
     * @return size_t
     */
    [[nodiscard]] auto getSize() const -> size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<size_t>>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<size_t> final {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        return std::vector<size_t>(wires.begin(), wires.end());
    }

    void applyInPlace(StateVectorManaged<T> &sv) const final {
        for (const auto &ob : obs_) {
            ob->applyInPlace(sv);
        }
    }

    [[nodiscard]] auto getObsName() const -> std::string final {
        using Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << "Observable: { 'name' : ";
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        obs_stream << ", 'wires' : " << getWires() << " }";
        return obs_stream.str();
    }
};

/// @cond DEV
namespace detail {

// Default implementation
template <class T, bool use_openmp> struct HamiltonianApplyInPlace {
    static void run(const std::vector<T> &coeffs,
                    const std::vector<std::shared_ptr<Observable<T>>> &terms,
                    StateVectorManaged<T> &sv) {
        StateVectorManaged<T> res(sv.getNumQubits());
        res.setZero();
        for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
            StateVectorManaged<T> tmp(sv);
            terms[term_idx]->applyInPlace(tmp);
            Util::scaleAndAdd(tmp.getLength(),
                              std::complex<T>{coeffs[term_idx], 0.0},
                              tmp.getData(), res.getData());
        }
        sv = std::move(res);
    }
};
#if defined(_OPENMP)
template <class T> struct HamiltonianApplyInPlace<T, true> {
    static void run(const std::vector<T> &coeffs,
                    const std::vector<std::shared_ptr<Observable<T>>> &terms,
                    StateVectorManaged<T> &sv) {
        const size_t length = sv.getLength();
#pragma omp parallel default(none) firstprivate(length)                        \
    shared(coeffs, terms, sv)
        {
            const auto nthreads = static_cast<size_t>(omp_get_num_threads());
            std::vector<std::complex<T>> local_sv(nthreads * length,
                                                  std::complex<T>{});

            int tid = omp_get_thread_num();

#pragma omp for
            for (size_t term_idx = 0; term_idx < terms.size(); term_idx++) {
                StateVectorManaged<T> tmp(sv);
                terms[term_idx]->applyInPlace(tmp);
                Util::scaleAndAdd(
                    length, std::complex<T>{coeffs[term_idx], 0.0},
                    tmp.getData(), local_sv.data() + length * tid);
            }

#pragma omp critical
            {
                sv.setZero();
                for (size_t i = 0; i < nthreads; i++) {
                    Util::scaleAndAdd(length, std::complex<T>{1.0, 0.0},
                                      local_sv.data() + length * i,
                                      sv.getData());
                }
            }
        }
    }
};
#endif

} // namespace detail
/// @endcond

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 * TODO: Check whether caching the sparse matrix representation can give
 * a speedup
 */
template <typename T> class Hamiltonian final : public Observable<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<T> coeffs_;
    std::vector<std::shared_ptr<Observable<T>>> obs_;

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param arg1 Arguments to construct coefficients
     * @param arg2 Arguments to construct observables
     */
    template <typename T1, typename T2>
    Hamiltonian(T1 &&arg1, T2 &&arg2)
        : coeffs_{std::forward<T1>(arg1)}, obs_{std::forward<T2>(arg2)} {}

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     */
    static auto
    create(std::initializer_list<T> arg1,
           std::initializer_list<std::shared_ptr<Observable<T>>> arg2)
        -> std::shared_ptr<Hamiltonian<T>> {
        return std::shared_ptr<Hamiltonian<T>>(
            new Hamiltonian<T>{std::move(arg1), std::move(arg2)});
    }

    void applyInPlace(StateVectorManaged<T> &sv) const final {
        detail::HamiltonianApplyInPlace<T, Util::Constant::use_openmp>::run(
            coeffs_, obs_, sv);
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> final {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        return std::vector<size_t>(wires.begin(), wires.end());
    }

    [[nodiscard]] auto getObsName() const -> std::string final {
        using Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_
           << "}, { 'observables' : ";
        const auto term_size = coeffs_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};

/**
 * @brief Utility class for encapsulating operations used by AdjointJacobian
 * class.
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

    /**
     * @brief Get total number of parameters.
     */
    [[nodiscard]] auto getTotalNumParams() const -> size_t {
        return std::accumulate(
            ops_params_.begin(), ops_params_.end(), size_t{0U},
            [](size_t acc, auto &params) { return acc + params.size(); });
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
    const std::vector<std::shared_ptr<Observable<T>>> observables;
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
     * calculation. This must be sorted.
     */
    JacobianData(size_t num_params, size_t num_elem, std::complex<T> *ps,
                 std::vector<std::shared_ptr<Observable<T>>> obs,
                 OpsData<T> ops, std::vector<size_t> trainP)
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
     * @return List of observables
     */
    [[nodiscard]] auto getObservables() const
        -> const std::vector<std::shared_ptr<Observable<T>>> & {
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
