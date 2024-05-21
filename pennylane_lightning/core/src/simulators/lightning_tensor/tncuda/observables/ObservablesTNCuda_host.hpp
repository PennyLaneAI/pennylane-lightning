// Copyright 2024 Xanadu Quantum Technologies Inc. and contributors.

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

#include <unordered_set>
#include <vector>

#include <iostream>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "TensorCuda.hpp"
#include "Util.hpp"

#include "cuError.hpp"
#include "cuGates_host.hpp"
#include "tncudaError.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;

template <class T> using vector1D = std::vector<T>;
template <class T> using vector2D = std::vector<std::vector<T>>;
template <class T> using vector3D = std::vector<std::vector<std::vector<T>>>;

} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {
/**
 * @brief A base class for observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class Observable {
  public:
    using CFP_t = typename StateTensorT::CFP_t;
    using PrecisionT = typename StateTensorT::PrecisionT;
    using ComplexT = typename StateTensorT::ComplexT;

  protected:
    vector1D<cuDoubleComplex> coeffs_; // coefficients of each term
    vector1D<std::size_t> numTensors_; // number of tensors in each term
    vector2D<std::size_t>
        numStateModes_; // number of state modes of each tensor in each term
    vector3D<std::size_t>
        stateModes_;       // state modes of each tensor in each term
    vector3D<CFP_t> data_; // data of each tensor in each term on host

  protected:
    Observable() = default;
    Observable(const Observable &) = default;
    Observable(Observable &&) noexcept = default;
    Observable &operator=(const Observable &) = default;
    Observable &operator=(Observable &&) noexcept = default;

  public:
    virtual ~Observable() = default;

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] virtual auto getObsName() const -> std::string = 0;

    /**
     * @brief Get the wires the observable applies to.
     */
    [[nodiscard]] virtual auto getWires() const -> std::vector<std::size_t> = 0;

    /**
     * @brief Get the observable data.
     *
     */
    [[nodiscard]] virtual auto getObs() const
        -> std::vector<std::shared_ptr<Observable<StateTensorT>>> {
        return {};
    };

    /**
     * @brief Apply the observable to the state tensor.
     *
     * @param sv State tensor to apply the observable to.
     */

    [[nodiscard]] auto getNumTensors() const -> const vector1D<std::size_t> & {
        return numTensors_;
    }

    /**
     * @brief Get the number of state modes of each tensor in each term.
     */
    [[nodiscard]] auto getNumStateModes() const
        -> const vector2D<std::size_t> & {
        return numStateModes_;
    }

    /**
     * @brief Get the state modes of each tensor in each term.
     */
    [[nodiscard]] auto getStateModes() const -> const vector3D<std::size_t> & {
        return stateModes_;
    }

    /**
     * @brief Get the data of each tensor in each term on host.
     */
    [[nodiscard]] auto getData() const -> const vector3D<CFP_t> & {
        return data_;
    }

    /**
     * @brief Get the data of each tensor in each term on host.
     */
    [[nodiscard]] auto getData() -> vector3D<CFP_t> & { return data_; }

    /**
     * @brief Get the coefficients of a observable.
     */
    [[nodiscard]] auto getCoeffs() const -> const vector1D<cuDoubleComplex> & {
        return coeffs_;
    };
};

/**
 * @brief Base class for named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT> class NamedObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;

  private:
    std::string obs_name_;
    std::vector<std::size_t> wires_;
    std::vector<PrecisionT> params_;

  public:
    /**
     * @brief Construct a NamedObsBase object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<std::size_t> wires,
             std::vector<PrecisionT> params = {})
        : obs_name_{obs_name}, wires_{wires}, params_{params} {
        this->coeffs_.push_back(cuDoubleComplex{1.0, 0.0});
        this->numTensors_.push_back(std::size_t{1});
        this->numStateModes_.push_back(
            vector1D<std::size_t>(std::size_t{1}, wires.size()));
        this->stateModes_.push_back(
            vector2D<std::size_t>(std::size_t{1}, wires));
        auto gateData =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance()
                .getGateData(obs_name, params);
        this->data_.push_back(vector2D<CFP_t>(std::size_t{1}, gateData));
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }
};

/**
 * @brief Base class for Hermitian observables
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT>
class HermitianObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;
    using ComplexT = typename StateTensorT::ComplexT;
    using MatrixT = std::vector<ComplexT>;

  private:
    MatrixT matrix_;
    std::vector<std::size_t> wires_;

  public:
    /**
     * @brief Construct a HermitianObs object, representing a given observable.
     *
     * @param matrix Matrix in row major format.
     * @param wires Wires the observable applies to.
     */
    HermitianObs(MatrixT matrix, std::vector<std::size_t> wires)
        : matrix_{std::move(matrix)}, wires_{std::move(wires)} {
        this->coeffs_.push_back(cuDoubleComplex{1.0, 0.0});
        this->numTensors_.push_back(std::size_t{1});
        this->numStateModes_.push_back(
            vector1D<std::size_t>(std::size_t{1}, wires_.size()));
        this->stateModes_.push_back(
            vector2D<std::size_t>(std::size_t{1}, wires_));
        // Convert matrix to vector of vector
        std::vector<CFP_t> matrix_cu(matrix_.size());
        std::transform(matrix_.begin(), matrix_.end(), matrix_cu.begin(),
                       [](const std::complex<PrecisionT> &x) {
                           return cuUtil::complexToCu<std::complex<PrecisionT>>(
                               x);
                       });
        this->data_.push_back(vector2D<CFP_t>(std::size_t{1}, matrix_cu));
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        return "Hermitian";
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }
};

/**
 * @brief Base class for a tensor product of observables.
 *
 * @tparam StateTensorT State tensor class.
 */
template <class StateTensorT>
class TensorProdObs : public Observable<StateTensorT> {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;
    using CFP_t = typename StateTensorT::CFP_t;

  protected:
    std::vector<std::shared_ptr<Observable<StateTensorT>>> obs_;
    std::vector<std::size_t> all_wires_;

  public:
    /**
     * @brief Create a tensor product of observables
     *
     * @param arg Arguments perfect forwarded to vector of observables.
     */
    template <typename... Ts>
    explicit TensorProdObs(Ts &&...arg) : obs_{std::forward<Ts>(arg)...} {
        if (obs_.size() == 1 &&
            obs_[0]->getObsName().find('@') != std::string::npos) {
            // This would prevent the misuse of this constructor for creating
            // TensorProdObs(TensorProdObs).
            PL_ABORT("A new TensorProdObs observable cannot be created "
                     "from a single TensorProdObs.");
        }

        this->coeffs_.push_back(cuDoubleComplex{1.0, 0.0});
        this->numTensors_.push_back(obs_.size());

        vector1D<std::size_t> numStateModesLocal;
        vector2D<std::size_t> stateModesLocal;
        vector2D<CFP_t> dataLocal;
        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            numStateModesLocal.emplace_back(ob_wires.size());
            stateModesLocal.emplace_back(ob_wires);
            dataLocal.emplace_back(ob->getData().front().front());
        }

        this->numStateModes_.emplace_back(numStateModesLocal);
        this->stateModes_.emplace_back(stateModesLocal);
        this->data_.emplace_back(dataLocal);

        std::unordered_set<std::size_t> wires;
        for (const auto &ob : obs_) {
            const auto ob_wires = ob->getWires();
            for (const auto wire : ob_wires) {
                PL_ABORT_IF(wires.contains(wire),
                            "All wires in observables must be disjoint.");
                wires.insert(wire);
            }
        }
        all_wires_ = std::vector<std::size_t>(wires.begin(), wires.end());
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
     * @return std::shared_ptr<TensorProdObs<StateTensorT>>
     */
    static auto
    create(std::initializer_list<std::shared_ptr<Observable<StateTensorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateTensorT>> {
        return std::shared_ptr<TensorProdObs<StateTensorT>>{
            new TensorProdObs(std::move(obs))};
    }

    /**
     * @brief Convenient wrapper for the constructor as the constructor does not
     * convert the std::shared_ptr with a derived class correctly.
     *
     * This function is useful as std::make_shared does not handle
     * brace-enclosed initializer list correctly.
     *
     * @param obs List of observables
     * @return std::shared_ptr<TensorProdObsBase<StateTensorT>>
     */
    static auto
    create(std::vector<std::shared_ptr<Observable<StateTensorT>>> obs)
        -> std::shared_ptr<TensorProdObs<StateTensorT>> {
        return std::shared_ptr<TensorProdObs<StateTensorT>>{
            new TensorProdObs(std::move(obs))};
    }

    /**
     * @brief Get the number of operations in observable.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getSize() const -> std::size_t { return obs_.size(); }

    /**
     * @brief Get the wires for each observable operation.
     *
     * @return const std::vector<std::vector<std::size_t>>&
     */
    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return all_wires_;
    }

    /**
     * @brief Get the observables in the tensor product.
     *
     * @return std::vector<std::shared_ptr<Observable<StateTensorT>>>
     */
    [[nodiscard]] auto getObs() const
        -> std::vector<std::shared_ptr<Observable<StateTensorT>>> override {
        return obs_;
    }

    /**
     * @brief Get the name of the observable
     */
    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
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

} // namespace Pennylane::LightningTensor::TNCuda::Observables
