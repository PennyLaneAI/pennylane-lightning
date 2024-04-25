// Copyright 2022-2024 Xanadu Quantum Technologies Inc. and contributors.

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

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "LinearAlg.hpp"
#include "Util.hpp"
#include "cuError.hpp"
#include "cuGateTensorCache.hpp"
#include "cuTensorNetError.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::Util;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::Observables {

/**
 * @brief A base class for all observable classes.
 *
 * We note that all subclasses must be immutable (does not provide any setter).
 *
 * @tparam T Floating point type
 */

template <typename T> class ObservableMPSCutn {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(T{}));

  private:
    /**
     * @brief Polymorphic function comparing this to another Observable
     * object.
     *
     * @param Another instance of subclass of Observable<T> to compare
     */
    [[nodiscard]] virtual bool
    isEqual(const ObservableMPSCutn<T> &other) const = 0;

  protected:
    ObservableMPSCutn() = default;
    ObservableMPSCutn(const ObservableMPSCutn &) = default;
    ObservableMPSCutn(ObservableMPSCutn &&) noexcept = default;
    ObservableMPSCutn &operator=(const ObservableMPSCutn &) = default;
    ObservableMPSCutn &operator=(ObservableMPSCutn &&) noexcept = default;

  public:
    virtual ~ObservableMPSCutn() = default;

    virtual void
    createTNOperator(const cutensornetHandle_t handle, cudaDataType_t typeData,
                     size_t numQubits, std::vector<size_t> qubitDims,
                     std::shared_ptr<GateTensorCache<T>> gateTensorCache) = 0;

    virtual cutensornetNetworkOperator_t getTNOperator() = 0;

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
    [[nodiscard]] bool operator==(const ObservableMPSCutn<T> &other) const {
        return typeid(*this) == typeid(other) && isEqual(other);
    }

    /**
     * @brief Test whether this object is different from another object.
     */
    [[nodiscard]] bool operator!=(const ObservableMPSCutn<T> &other) const {
        return !(*this == other);
    }
};

/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam TensorNetT State tensor class.
 */
template <typename T>
class NamedObsMPSCutn final : public ObservableMPSCutn<T> {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(T{}));

  private:
    std::string obs_name_;
    std::vector<size_t> wires_;
    std::vector<T> params_;

    [[nodiscard]] bool
    isEqual(const ObservableMPSCutn<T> &other) const override {
        const auto &other_cast = static_cast<const NamedObsMPSCutn<T> &>(other);

        return (obs_name_ == other_cast.obs_name_) &&
               (wires_ == other_cast.wires_) && (params_ == other_cast.params_);
    }

  private:
    cutensornetNetworkOperator_t obsOperator_{nullptr};
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorData_;
    int64_t id_;

  public:
    /**
     * @brief Construct a NamedObs object, representing a given observable.
     *
     * @param obs_name Name of the observable.
     * @param wires Argument to construct wires.
     * @param params Argument to construct parameters
     */
    NamedObsMPSCutn(std::string obs_name, std::vector<size_t> wires,
                    std::vector<T> params = {})
        : obs_name_{std::move(obs_name)}, wires_{std::move(wires)},
          params_{std::move(params)} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());

        wires_int_.resize(wires_.size());

        std::transform(wires_.begin(), wires_.end(), wires_int_.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        numStateModes_.push_back(static_cast<int32_t>(wires_.size()));

        stateModes_.push_back(wires_int_.data());

        tensorData_.push_back(nullptr);
    }

    ~NamedObsMPSCutn() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(obsOperator_));
    }

    [[nodiscard]] auto getObsName() const -> std::string override {
        using Pennylane::Util::operator<<;
        std::ostringstream obs_stream;
        obs_stream << obs_name_ << wires_;
        return obs_stream.str();
    }

    [[nodiscard]] auto getWires() const -> std::vector<size_t> override {
        return wires_;
    }

    void createTNOperator(const cutensornetHandle_t handle,
                          cudaDataType_t typeData, size_t numQubits,
                          std::vector<size_t> qubitDims,
                          std::shared_ptr<GateTensorCache<T>> gateTensorCache) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ handle,
            /* int32_t */ static_cast<int32_t>(numQubits),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(qubitDims.data()),
            /* cudaDataType_t */ typeData,
            /* cutensornetNetworkOperator_t */ &obsOperator_));

        auto &&par = (params_.empty()) ? std::vector<T>{0.0} : params_;
        // TODO check if obs exists.
        tensorData_[0] = static_cast<const void *>(
            gateTensorCache->get_gate_device_ptr(obs_name_, par[0]));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ handle,
            /* cutensornetNetworkOperator_t */ obsOperator_,
            /* cuDoubleComplex coefficient*/ cuDoubleComplex{1, 0.0},
            /* int32_t numTensors */ 1,
            /* const int32_t numStateModes[] */ numStateModes_.data(),
            /* const int32_t *stateModes[] */ stateModes_.data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorData_.data(),
            /* int64_t* */ &id_));
    }

    cutensornetNetworkOperator_t getTNOperator() { return obsOperator_; }
};

} // namespace Pennylane::LightningTensor::Observables
