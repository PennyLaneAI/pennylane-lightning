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

#include <cutensornet.h>
#include <functional>
#include <vector>

#include <iostream>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Observables.hpp"
#include "TNCudaGateCache.hpp"
#include "TensorCuda.hpp"
#include "Util.hpp"
#include "cuError.hpp"
#include "tncudaError.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {

/**
 * @brief A base class (CRTP) for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam StateTensorT State vector class.
 */
template <class StateTensorT> class ObservableTNCuda {
  public:
    using PrecisionT = typename StateTensorT::PrecisionT;

  protected:
    ObservableTNCuda() = default;
    ObservableTNCuda(const ObservableTNCuda &) = default;
    ObservableTNCuda(ObservableTNCuda &&) noexcept = default;
    ObservableTNCuda &operator=(const ObservableTNCuda &) = default;
    ObservableTNCuda &operator=(ObservableTNCuda &&) noexcept = default;

  protected:
    // cutensornetNetworkOperator_t obsOperator_{nullptr};

    /**
     * @brief Apply the observable to the given statevector in place.
     */
    /*
    void createTNOperator(StateTensorT &state_tensor) {
        // PL_ABORT_IF_NOT(
        //     obsOperator_ == nullptr,
        //     "The createTNOperator() method can be called only once.");

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /-* const cutensornetHandle_t *-/ state_tensor.getTNCudaHandle(),
            /-* int32_t *-/ static_cast<int32_t>(state_tensor.getNumQubits()),
            /-* const int64_t stateModeExtents *-/
            reinterpret_cast<int64_t *>(
                const_cast<size_t *>(state_tensor.getQubitDims().data())),
            /-* cudaDataType_t *-/ state_tensor.getCudaDataType(),
            /-* cutensornetNetworkOperator_t *-/ &obsOperator_));
    }
    */

  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<StateVectorT> to
     * compare.
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableTNCuda<StateTensorT> &other) const = 0;

  public:
    virtual ~ObservableTNCuda() {
        // if (obsOperator_ != nullptr) {
        //     PL_CUTENSORNET_IS_SUCCESS(
        //         cutensornetDestroyNetworkOperator(obsOperator_));
        // }
    }

    // cutensornetNetworkOperator_t getTNOperator() { return obsOperator_; }

    virtual void appendTNOperator(StateTensorT &state_tensor) = 0;

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
        -> std::vector<std::shared_ptr<ObservableTNCuda<StateTensorT>>> {
        return {};
    };

    /**
     * @brief Get the coefficients of a Hamiltonian observable.
     */
    [[nodiscard]] virtual auto getCoeffs() const -> std::vector<PrecisionT> {
        return {};
    };

    /**
     * @brief Test whether this object is equal to another object
     */
    [[nodiscard]] auto operator==(const Observable<StateTensorT> &other) const
        -> bool {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] auto operator!=(const Observable<StateTensorT> &other) const
        -> bool {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <typename StateTensorT>
class NamedObs final : public ObservableTNCuda<StateTensorT> {
  private:
    using BaseType = ObservableTNCuda<StateTensorT>;
    using PrecisionT = typename StateTensorT::PrecisionT;

  private:
    std::string obs_name_;
    std::vector<std::size_t> wires_;
    std::vector<PrecisionT> params_;

  private:
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorDataPtr_;
    std::vector<TensorCuda<PrecisionT>> tensorData_;

    int64_t id_;

  private:
    [[nodiscard]] auto
    isEqual(const ObservableTNCuda<StateTensorT> &other) const
        -> bool override {
        const auto &other_cast =
            static_cast<const NamedObs<StateTensorT> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  public:
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObs(std::string obs_name, std::vector<std::size_t> wires,
             std::vector<PrecisionT> params = {})
        : obs_name_{std::move(obs_name)}, wires_{wires},
          params_{std::move(params)} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());

        wires_int_ = std::vector<int32_t>(wires_.size());

        std::transform(wires_.begin(), wires_.end(), wires_int_.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        numStateModes_.push_back(static_cast<int32_t>(wires_.size()));

        stateModes_.push_back(wires_int_.data());
    }

    ~NamedObs() {}

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<std::size_t> override {
        return wires_;
    }

    void appendTNOperator(StateTensorT &state_tensor) {
        // this->createTNOperator(state_tensor);

        auto &&par = (params_.empty()) ? std::vector<PrecisionT>{0.0} : params_;

        std::size_t rank = 2 * wires_int_.size();
        std::vector<std::size_t> modes(rank, 0);
        std::vector<std::size_t> extents(rank, 2);

        tensorData_.emplace_back(rank, modes, extents,
                                 state_tensor.getDevTag());

        auto gate_host_vector =
            state_tensor.getGateCache()->get_gate_host_vector(obs_name_, par);

        tensorData_.back().getDataBuffer().CopyHostDataToGpu(
            gate_host_vector.data(), gate_host_vector.size());

        tensorDataPtr_.push_back(tensorData_.back().getDataBuffer().getData());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ state_tensor.getTNCudaHandle(),
            /* cutensornetNetworkOperator_t */ state_tensor.getTNOperator(),
            /* cuDoubleComplex coefficient*/ cuDoubleComplex{1, 0.0},
            /* int32_t numTensors */ 1,
            /* const int32_t numStateModes[] */ numStateModes_.data(),
            /* const int32_t *stateModes[] */ stateModes_.data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorDataPtr_.data(),
            /* int64_t* */ &id_));
    }
};
} // namespace Pennylane::LightningTensor::TNCuda::Observables
