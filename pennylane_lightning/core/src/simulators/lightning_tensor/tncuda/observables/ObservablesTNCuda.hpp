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

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "Observables.hpp"
#include "TNCudaGateCache.hpp"
#include "Util.hpp"
#include "cuError.hpp"
#include "tncudaError.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor;
using namespace Pennylane::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Observables {
/**
 * @brief Class models named observables (PauliX, PauliY, PauliZ, etc.)
 *
 * @tparam StateVectorT State vector class.
 */
template <typename StateTensorT>
class NamedObs final : public NamedObsBase<StateTensorT> {
  private:
    using BaseType = NamedObsBase<StateTensorT>;
    using PrecisionT = typename StateTensorT::PrecisionT;

  private:
    cutensornetNetworkOperator_t obsOperator_{nullptr};
    std::vector<int32_t> wires_int_;
    std::vector<int32_t> numStateModes_;
    std::vector<const int32_t *> stateModes_;
    std::vector<const void *> tensorDataPtr_;
    std::vector<TensorCuda<PrecisionT>> tensorData_;

    int64_t id_;

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
        : BaseType{obs_name, wires, params} {
        using Pennylane::Gates::Constant::gate_names;
        using Pennylane::Gates::Constant::gate_num_params;
        using Pennylane::Gates::Constant::gate_wires;

        const auto gate_op = lookup(reverse_pairs(gate_names),
                                    std::string_view{this->obs_name_});
        PL_ASSERT(lookup(gate_wires, gate_op) == this->wires_.size());
        PL_ASSERT(lookup(gate_num_params, gate_op) == this->params_.size());

        wires_int_.resize(wires.size());

        std::transform(wires.begin(), wires.end(), wires_int_.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        numStateModes_.push_back(static_cast<int32_t>(wires.size()));

        stateModes_.push_back(wires_int_.data());
    }

    ~NamedObs() {
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(obsOperator_));
    }

    void createTNOperator(const StateTensorT &state_tensor) {

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ state_tensor.getTNCudaHandle(),
            /* int32_t */ static_cast<int32_t>(state_tensor.getNumQubits()),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(
                const_cast<size_t *>(state_tensor.getQubitDims().data())),
            /* cudaDataType_t */ state_tensor.getCudaDataType(),
            /* cutensornetNetworkOperator_t */ &obsOperator_));

        auto &&par = (this->params_.empty()) ? std::vector<PrecisionT>{0.0}
                                             : this->params_;

        std::size_t rank = size_t{2} * wires_int_.size();
        std::vector<std::size_t> modes(rank, 0);
        std::vector<std::size_t> extents(rank, 2);

        auto &&tensor = TensorCuda<PrecisionT>(rank, modes, extents,
                                               state_tensor.getDevTag());

        auto &&gate_host_vector =
            state_tensor.getGateCache()->get_gate_host_vector(this->obs_name_,
                                                              par);

        tensor.getDataBuffer().CopyHostDataToGpu(gate_host_vector.data(),
                                                 gate_host_vector.size());

        tensorData_.emplace_back(std::move(tensor));

        tensorDataPtr_[0] = tensorData_[0].getDataBuffer().getData();

        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ state_tensor.getTNCudaHandle(),
            /* cutensornetNetworkOperator_t */ obsOperator_,
            /* cuDoubleComplex coefficient*/ cuDoubleComplex{1, 0.0},
            /* int32_t numTensors */ 1,
            /* const int32_t numStateModes[] */ numStateModes_.data(),
            /* const int32_t *stateModes[] */ stateModes_.data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorDataPtr_.data(),
            /* int64_t* */ &id_));
    }

    cutensornetNetworkOperator_t getTNOperator() { return obsOperator_; }

    void applyInPlace([[maybe_unused]] StateTensorT &sv) const override {
        PL_ABORT("Lightning.Tensor doesn't support the applyInPlace() method.");
    }

    void applyInPlaceShots(
        [[maybe_unused]] StateTensorT &sv,
        [[maybe_unused]] std::vector<std::vector<PrecisionT>> &eigenValues,
        [[maybe_unused]] std::vector<std::size_t> &ob_wires) const override {
        PL_ABORT(
            "Lightning.Tensor doesn't support the applyInPlaceShots() method.");
    }
};
} // namespace Pennylane::LightningTensor::TNCuda::Observables
