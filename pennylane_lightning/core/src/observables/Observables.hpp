// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
#include <complex>
#include <memory>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "Error.hpp"
#include "Util.hpp"

namespace Pennylane::Observables {

/**
 * @brief A base class (CRTP) for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT> class Observable {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    Observable() = default;
    Observable(const Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(const Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<StateVectorT> to
     * compare.
     */
    [[nodiscard]] virtual bool
    isEqual(const Observable<StateVectorT> &other) const = 0;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    virtual void applyInPlace(StateVectorT &sv) const = 0;

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
    [[nodiscard]] auto operator==(const Observable<StateVectorT> &other) const
        -> bool {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] auto operator!=(const Observable<StateVectorT> &other) const
        -> bool {
        return !(*this == other);
    }
};

/**
 * @brief Base class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class NamedObsBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<PrecisionT> params_;

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const NamedObsBase<StateVectorT> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObsBase object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsBase(std::string obs_name, std::vector<size_t> wires,
                 std::vector<PrecisionT> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    void applyInPlace(StateVectorT &sv) const override {
        sv.applyOperation(obs_name_, wires_, false, params_);
    }
};

/**
 * @brief Base class for Hermitian observables
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HermitianObsBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

  protected:
    MatrixT matrix_;
    std::vector<size_t> wires_;

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const HermitianObsBase<StateVectorT> &>(other);

        return (matrix_ == other_cast.matrix_) && (wires_ == other_cast.wires_);
    }

  public:
    /**
     * @brief Create an Hermitian observable
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObsBase(MatrixT matrix, std::vector<size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {
        PL_ASSERT(matrix_.size() == Util::exp2(2 * wires_.size()));
    }

    [[nodiscard]] auto getMatrix() const -> const MatrixT & { return matrix_; }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    void applyInPlace(StateVectorT &sv) const override {
        sv.applyMatrix(matrix_, wires_);
    }
};

/**
 * @brief Base class for a tensor product of observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class TensorProdObsBase : public Observable<StateVectorT> {
  protected:
    std::vector<std::shared_ptr<Observable<StateVectorT>>> obs_;
    std::vector<size_t> all_wires_;

  private:
    [[nodiscard]] auto isEqual(const Observable<StateVectorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const TensorProdObsBase<StateVectorT> &>(other);

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
    explicit TensorProdObsBase(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        std::unordered_set<size_t> wires;

        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                PL_ABORT_IF(wires.contains(wire),
                            "All wires in observables must be disjoint.");
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
     * @return std::shared_ptr<TensorProdObsBase<StateVectorT>>
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsBase<StateVectorT>> {
        return std::shared_ptr<TensorProdObsBase<StateVectorT>>{
            new TensorProdObsBase(std::move(obs))};
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObsBase<StateVectorT>>
     */
    static auto
    create(std::vector<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<TensorProdObsBase<StateVectorT>> {
        return std::shared_ptr<TensorProdObsBase<StateVectorT>>{
            new TensorProdObsBase(std::move(obs))};
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

    void applyInPlace(StateVectorT &sv) const override {
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

/**
 * @brief Base class for a general Hamiltonian representation as a sum of
 * observables.
 *
 * @tparam StateVectorT State vector class.
 */
template <class StateVectorT>
class HamiltonianBase : public Observable<StateVectorT> {
  public:
    using PrecisionT = typename StateVectorT::PrecisionT;

  protected:
    std::vector<PrecisionT> coeffs_;
    std::vector<std::shared_ptr<Observable<StateVectorT>>> obs_;

  private:
    [[nodiscard]] bool
    isEqual(const Observable<StateVectorT> &other) const override {
        const auto &other_cast =
            static_cast<const HamiltonianBase<StateVectorT> &>(other);

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
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     */
    template <typename T1, typename T2>
    HamiltonianBase(T1 &&coeffs, T2 &&obs)
        : coeffs_{std::forward<T1>(coeffs)}, obs_{std::forward<T2>(obs)} {
        PL_ASSERT(coeffs_.size() == obs_.size());
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param coeffs Arguments to construct coefficients
     * @param obs Arguments to construct observables
     * @return std::shared_ptr<HamiltonianBase<StateVectorT>>
     */
    static auto
    create(std::initializer_list<PrecisionT> coeffs,
           std::initializer_list<std::shared_ptr<Observable<StateVectorT>>> obs)
        -> std::shared_ptr<HamiltonianBase<StateVectorT>> {
        return std::shared_ptr<HamiltonianBase<StateVectorT>>(
            new HamiltonianBase<StateVectorT>{std::move(coeffs),
                                              std::move(obs)});
    }

    void applyInPlace([[maybe_unused]] StateVectorT &sv) const override {
        PL_ABORT("For Hamiltonian Observables, the applyInPlace method must be "
                 "defined at the backend level.");
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

} // namespace Pennylane::Observables