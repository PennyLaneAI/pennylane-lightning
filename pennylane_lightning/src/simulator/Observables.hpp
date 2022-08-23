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
#pragma once

#include "Error.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"
#include "Util.hpp"

#include <memory>
#include <unordered_set>

namespace Pennylane::Simulators {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */
template <typename T> class Observable {
  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool isEqual(const Observable<T> &other) const = 0;

  protected:
    Observable() = default;
    Observable(const Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(const Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorManagedCPU<T> &sv) const = 0;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<size_t> = 0;

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] bool operator==(const Observable<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const Observable<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam T Floating point type
 */
template <typename T> class NamedObs final : public Observable<T> {
  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

    [[nodiscard]] bool isEqual(const Observable<T> &other) const override {
        const auto &other_cast = static_cast<const NamedObs<T> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param arg1 Name of the observable.
     * @param arg2 Argument to construct wires.
     * @param arg3 Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<size_t> wires,
             std::vector<T> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {
        using Gates::Constant::gate_names;
        using Gates::Constant::gate_num_params;
        using Gates::Constant::gate_wires;

        const auto gate_op = Util::lookup(Util::reverse_pairs(gate_names),
                                          std::string_view{obs_name_});
        PL_ASSERT(Util::lookup(gate_wires, gate_op) == wires_.size());
        PL_ASSERT(Util::lookup(gate_num_params, gate_op) == params_.size());
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    void applyInPlace(StateVectorManagedCPU<T> &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Class models
 *
 */
template <typename T> class HermitianObs final : public Observable<T> {
  public:
    using MatrixT = std::vector<std::complex<T>>;

  private:
    MatrixT matrix_;
    std::vector<size_t> wires_;

    [[nodiscard]] bool isEqual(const Observable<T> &other) const override {
        const auto &other_cast = static_cast<const HermitianObs<T> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    template <typename T1>
    HermitianObs(T1 &&matrix, std::vector<size_t> wires)
        : matrix_{std::forward<T1>(matrix)}, wires_{std::move(wires)} {
        PL_ASSERT(matrix_.size() ==
                  Util::exp2(wires_.size()) * Util::exp2(wires_.size()));
    }

    [[nodiscard]] auto getMatrix() const -> const MatrixT & { return matrix_; }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    void applyInPlace(StateVectorManagedCPU<T> &sv) const override {
        sv.applyMatrix(matrix_, wires_);
    }
};

/**
 * @brief Tensor product observable class
 */
template <typename T> class TensorProdObs final : public Observable<T> {
  private:
    std::vector<std::shared_ptr<Observable<T>>> obs_;
    std::vector<size_t> all_wires_;

    [[nodiscard]] bool isEqual(const Observable<T> &other) const override {
        const auto &other_cast = static_cast<const TensorProdObs<T> &>(other);

        if (obs_.size() != other_cast.obs_.size()) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                if (wires.contains(wire)) {
                    PL_ABORT("All wires in observables must be disjoint.");
                }
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires_.begin(), all_wires_.end());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<T>>> obs)
        -> std::shared_ptr<TensorProdObs<T>> {
        return std::shared_ptr<TensorProdObs<T>>{
            new TensorProdObs(std::move(obs))};
    }

    static auto create(std::vector<std::shared_ptr<Observable<T>>> obs)
        -> std::shared_ptr<TensorProdObs<T>> {
        return std::shared_ptr<TensorProdObs<T>>{
            new TensorProdObs(std::move(obs))};
    }

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
    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return all_wires_;
    }

    void applyInPlace(StateVectorManagedCPU<T> &sv) const override {
        for (const auto &ob : obs_) {
            ob->applyInPlace(sv);
        }
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream obs_stream;
        const auto obs_size = obs_.size();
        for (size_t idx = 0; idx < obs_size; idx++) {
            obs_stream << obs_[idx]->getObsName();
            if (idx != obs_size - 1) {
                obs_stream << " @ ";
            }
        }
        return obs_stream.str();
    }
};
/// @cond DEV
namespace detail {
// Default implementation
template <class T, bool use_openmp> struct HamiltonianApplyInPlace {
    static void run(const std::vector<T> &coeffs,
                    const std::vector<std::shared_ptr<Observable<T>>> &terms,
                    StateVectorManagedCPU<T> &sv) {
        auto allocator = sv.allocator();
        std::vector<std::complex<T>, decltype(allocator)> res(
            sv.getLength(), std::complex<T>{0.0, 0.0}, allocator);
        for (size_t term_idx = 0; term_idx < coeffs.size(); term_idx++) {
            StateVectorManagedCPU<T> tmp(sv);
            terms[term_idx]->applyInPlace(tmp);
            Util::scaleAndAdd(tmp.getLength(),
                              std::complex<T>{coeffs[term_idx], 0.0},
                              tmp.getData(), res.data());
        }
        sv.updateData(res);
    }
};
#if defined(_OPENMP)
template <class T> struct HamiltonianApplyInPlace<T, true> {
    static void run(const std::vector<T> &coeffs,
                    const std::vector<std::shared_ptr<Observable<T>>> &terms,
                    StateVectorManagedCPU<T> &sv) {
        const size_t length = sv.getLength();
        const auto allocator = sv.allocator();

        std::vector<std::complex<T>, decltype(allocator)> sum(
            length, std::complex<T>{}, allocator);

#pragma omp parallel default(none) firstprivate(length, allocator)             \
    shared(coeffs, terms, sv, sum)
        {
            StateVectorManagedCPU<T> tmp(sv.getNumQubits());

            std::vector<std::complex<T>, decltype(allocator)> local_sv(
                length, std::complex<T>{}, allocator);

#pragma omp for
            for (size_t term_idx = 0; term_idx < terms.size(); term_idx++) {
                tmp.updateData(sv.getDataVector());
                terms[term_idx]->applyInPlace(tmp);
                Util::scaleAndAdd(length,
                                  std::complex<T>{coeffs[term_idx], 0.0},
                                  tmp.getData(), local_sv.data());
            }

#pragma omp critical
            {
                Util::scaleAndAdd(length, std::complex<T>{1.0, 0.0},
                                  local_sv.data(), sum.data());
            }
        }

        sv.updateData(sum);
    }
};
#endif

} // namespace detail
/// @endcond

/**
 * @brief General Hamiltonian as a sum of observables.
 *
 * TODO: Check whether caching a sparse matrix representation can give
 * a speedup
 */
template <typename T> class Hamiltonian final : public Observable<T> {
  public:
    using PrecisionT = T;

  private:
    std::vector<T> coeffs_;
    std::vector<std::shared_ptr<Observable<T>>> obs_;

    [[nodiscard]] bool isEqual(const Observable<T> &other) const override {
        const auto &other_cast = static_cast<const Hamiltonian<T> &>(other);

        if (coeffs_ != other_cast.coeffs_) {
            return false;
        }

        for (size_t i = 0; i < obs_.size(); i++) {
            if (*obs_[i] != *other_cast.obs_[i]) {
                return false;
            }
        }
        return true;
    }

  public:
    /**
     * @brief Create a Hamiltonian from coefficients and observables
     *
     * @param arg1 Arguments to construct coefficients
     * @param arg2 Arguments to construct observables
     */
    template <typename T1, typename T2>
    Hamiltonian(T1 &&arg1, T2 &&arg2)
        : coeffs_{std::forward<T1>(arg1)}, obs_{std::forward<T2>(arg2)} {
        PL_ASSERT(coeffs_.size() == obs_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param arg1 Argument to construct coefficients
     * @param arg2 Argument to construct terms
     */
    static auto
    create(std::initializer_list<T> arg1,
           std::initializer_list<std::shared_ptr<Observable<T>>> arg2)
        -> std::shared_ptr<Hamiltonian<T>> {
        return std::shared_ptr<Hamiltonian<T>>(
            new Hamiltonian<T>{std::move(arg1), std::move(arg2)});
    }

    void applyInPlace(StateVectorManagedCPU<T> &sv) const override {
        detail::HamiltonianApplyInPlace<T, Util::Constant::use_openmp>::run(
            coeffs_, obs_, sv);
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            wires.insert(ob_wires.begin(), ob_wires.end());
        }
        auto all_wires = std::vector<size_t>(wires.begin(), wires.end());
        std::sort(all_wires.begin(), all_wires.end());
        return all_wires;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream ss;
        ss << "Hamiltonian: { 'coeffs' : " << coeffs_ << ", 'observables' : [";
        const auto term_size = coeffs_.size();
        for (size_t t = 0; t < term_size; t++) {
            ss << obs_[t]->getObsName();
            if (t != term_size - 1) {
                ss << ", ";
            }
        }
        ss << "]}";
        return ss.str();
    }
};

} // namespace Pennylane::Simulators
