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
} // namespace Pennylane::LightningTensor::TNCuda::Observables
