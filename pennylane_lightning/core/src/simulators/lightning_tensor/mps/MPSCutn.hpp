// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file MPSCutn.hpp
 * cuTensorNetwork backed MPS class.
 */

#pragma once

#include <complex>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <cuda.h>
#include <cutensornet.h>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "MPSCutnBase.hpp"
#include "ObservablesMPSCutn.hpp"
#include "TensorBase.hpp"
#include "cuDeviceTensor.hpp"
#include "cuGateTensorCache.hpp"
#include "cuTensorNetError.hpp"
#include "cuTensorNet_helpers.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::Util;
using namespace Pennylane::LightningTensor::Observables;

// Function to convert a size_t value to a binary string
std::string size_t_to_binary_string(const size_t &numQubits, size_t val) {
    std::string str;
    for (size_t i = 0; i < numQubits; i++) {
        str = (val >> i) & 1 ? ('1' + str) : ('0' + str);
    }
    return str;
}

// Get scratch memory size
std::size_t getScratchMemorySize() {
    std::size_t freeBytes{0}, totalBytes{0};
    PL_CUDA_IS_SUCCESS(cudaMemGetInfo(&freeBytes, &totalBytes));
    std::size_t scratchSize = (freeBytes - (totalBytes % 4096)) / 2;
    return scratchSize;
}
} // namespace
/// @endcond

namespace Pennylane::LightningTensor {

template <class Precision>
class MPSCutn : public MPSCutnBase<Precision, MPSCutn<Precision>> {
  private:
    using BaseType = MPSCutnBase<Precision, MPSCutn<Precision>>;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  public:
    // TODO add SVD options by the cutensornetStateConfigure() API
    //  cutensornetStateAttributes_t,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO
    MPSCutn(size_t numQubits, size_t maxExtent, std::vector<size_t> qubitDims,
            DevTag<int> dev_tag)
        : BaseType(numQubits, maxExtent, qubitDims, dev_tag) {}

    MPSCutn(const MPSCutn &other)
        : BaseType(other.getNumQubits(), other.getMaxExtent(),
                   other.getQubitDims(), other.getDevTag()) {

        BaseType::CopyGpuDataToGpuIn(other);

        updateMPSTensorData_(BaseType::getSiteExtentsPtr().data(),
                             BaseType::getMPSTensorDataPtr().data());
    }

    ~MPSCutn() {}

    // Set a zero state for d_mpsTensors
    void reset() {
        size_t index = 0;
        this->setBasisState(index);
    }

    // TODO this implementation only support up to 64 qubits
    // TODO SWITCH TO std::vector<char or size_t> index to accept basis state
    // array (list) from Pennylane layer
    void setBasisState(size_t index) {
        // Assuming the site vector is [1,0] or [0,1] and bond vector is
        // [1,0,0...].
        std::string str =
            size_t_to_binary_string(BaseType::getNumQubits(), index);

        CFP_t value_cu =
            Pennylane::LightningGPU::Util::complexToCu<std::complex<Precision>>(
                {1.0, 0.0});

        for (size_t i = 0; i < BaseType::getNumQubits(); i++) {
            BaseType::getMPSTensorData()[i].getDataBuffer().zeroInit();

            size_t target = 0;

            if (i == 0) {
                target =
                    str.at(BaseType::getNumQubits() - 1 - i) == '0' ? 0 : 1;
            } else if (i == BaseType::getNumQubits() - 1) {
                target = str.at(BaseType::getNumQubits() - 1 - i) == '0'
                             ? 0
                             : BaseType::getMaxExtent();
            } else {
                target = str.at(BaseType::getNumQubits() - 1 - i) == '0'
                             ? 0
                             : BaseType::getMaxExtent();
            }

            PL_CUDA_IS_SUCCESS(cudaMemcpy(&BaseType::getMPSTensorData()[i]
                                               .getDataBuffer()
                                               .getData()[target],
                                          &value_cu, sizeof(CFP_t),
                                          cudaMemcpyHostToDevice));
        }

        updateMPSTensorData_(BaseType::getSiteExtentsPtr().data(),
                             BaseType::getMPSTensorDataPtr().data());
    };

    auto getDataVector() -> std::vector<std::complex<Precision>> {
        // 1D representation of mpsTensor
        std::vector<size_t> modes(1, 1);
        std::vector<size_t> extent(1, (1 << BaseType::getNumQubits()));
        cuDeviceTensor<Precision> d_mpsTensor(modes.size(), modes, extent,
                                              BaseType::getDevTag());

        std::vector<void *> d_mpsTensorsPtr(
            1, static_cast<void *>(d_mpsTensor.getDataBuffer().getData()));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            BaseType::getCutnHandle(), &workDesc));

        const std::size_t scratchSize = getScratchMemorySize();

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused in v24.03*/ 0x0));

        int64_t worksize = this->getWorkSpaceMemorySize_(workDesc);

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<size_t, int> d_scratch(d_scratch_length,
                                          BaseType::getDevTag(), true);

        this->setWorkSpaceMemory_(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        std::vector<int64_t *> extentsPtr;
        std::vector<int64_t> extent_int64(1, (1 << BaseType::getNumQubits()));
        extentsPtr.emplace_back(extent_int64.data());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr.data(),
            /* int64_t *stridesOut */ nullptr,
            /* void * */ d_mpsTensorsPtr.data(),
            /* cudaStream_t */ BaseType::getDevTag().getStreamID()));

        std::vector<ComplexT> results(extent.front());

        d_mpsTensor.CopyGpuDataToHost(results.data(), results.size());

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        return results;
    }

    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<Precision> &params = {0.0},
                        const std::vector<CFP_t> &gate_matrix = {}) {
        PL_ABORT_IF(wires.size() > 2,
                    "Current version only supports 1/2 qubit gates.");

        const auto ctrl_offset = (BaseType::getCtrlMap().find(opName) !=
                                  BaseType::getCtrlMap().end())
                                     ? BaseType::getCtrlMap().at(opName)
                                     : 0;
        const std::vector<std::size_t> ctrls{wires.begin(),
                                             wires.begin() + ctrl_offset};
        const std::vector<std::size_t> tgts{wires.begin() + ctrl_offset,
                                            wires.end()};

        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;

        if (opName == "Identity") {
            return;
        } else {
            if (!BaseType::getGateCache()->gateExists(opName, par[0]) &&
                gate_matrix.empty()) {
                std::string message = "Currently unsupported gate: " + opName;
                throw LightningException(message);
            } else if (!BaseType::getGateCache()->gateExists(opName, par[0])) {
                BaseType::getGateCache()->add_gate(opName, par[0], gate_matrix);
            }

            if (ctrls.size() > 0) {
                applyControlledGate_(
                    BaseType::getGateCache()->get_gate_device_ptr(opName,
                                                                  par[0]),
                    ctrls, tgts, adjoint);
            } else {
                applyGate_(BaseType::getGateCache()->get_gate_device_ptr(
                               opName, par[0]),
                           wires, adjoint);
            }
        }
    }

    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<size_t>> &ops_wires,
                    const std::vector<bool> &ops_adjoint,
                    const std::vector<std::vector<Precision>> &ops_params) {
        const size_t numOperations = ops.size();
        PL_ABORT_IF(
            numOperations != ops_wires.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_adjoint.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_params.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            this->applyOperation(ops[i], ops_wires[i], ops_adjoint[i],
                                 ops_params[i]);
        }
    }

    /**
     * @brief Apply multiple gates to the state-tensor.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<size_t>> &ops_wires,
                         const std::vector<bool> &ops_adjoint) {
        const size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (size_t i = 0; i < numOperations; i++) {
            this->applyOperation(ops[i], ops_wires[i], ops_adjoint[i], {});
        }
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param adjoint Indicate whether inverse should be taken.
     */
    void applyMatrix(const std::complex<PrecisionT> *gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::string opName = {};
        size_t n = size_t{1} << wires.size();
        const std::vector<std::complex<PrecisionT>> matrix(gate_matrix,
                                                           gate_matrix + n * n);
        std::vector<CFP_t> matrix_cu(matrix.size());
        std::transform(matrix.begin(), matrix.end(), matrix_cu.begin(),
                       [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, {}, matrix_cu);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * std vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param adjoint Indicate whether inverse should be taken.
     */
    void applyMatrix(const std::vector<std::complex<PrecisionT>> &gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(gate_matrix.size() !=
                        Pennylane::Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(gate_matrix.data(), wires, adjoint);
    }

    // ComplexT expval(const std::string &opName, const std::vector<size_t>
    // &wires,
    //                 const std::vector<Precision> &params = {0.0}) {
    //     auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
    //     return expval_(
    //         BaseType::getGateCache()->get_gate_device_ptr(opName, par[0]),
    //         wires);
    // }

    ComplexT
    expval(Pennylane::LightningTensor::Observables::ObservableMPSCutn<Precision>
               &ob) {

        ob.createTNOperator(BaseType::getCutnHandle(), BaseType::getDataType(),
                            BaseType::getNumQubits(), BaseType::getQubitDims(),
                            BaseType::getGateCache());

        return expval_(ob.getTNOperator());
    }

  private:
    size_t getWorkSpaceMemorySize_(cutensornetWorkspaceDescriptor_t &workDesc) {
        int64_t worksize{0};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetWorksizePref_t */
            CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /*  int64_t * */ &worksize));

        return worksize;
    }

    void setWorkSpaceMemory_(cutensornetWorkspaceDescriptor_t &workDesc,
                             void *scratchPtr, int64_t &worksize) {

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceSetMemory(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /* void *const */ scratchPtr,
            /* int64_t */ worksize));
    }

    void updateMPSTensorData_(const int64_t *const *extentsIn,
                              void **stateTensorsIn) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t*/ BaseType::getCutnHandle(),
            /*cutensornetState_t*/ BaseType::getQuantumState(),
            /*cutensornetBoundaryCondition_t*/
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const*/ extentsIn,
            /*const int64_t *const*/ nullptr,
            /*void **/ stateTensorsIn));
    }

    void applyGate_(CFP_t *gateTensorPtr, const std::vector<size_t> &wires,
                    bool adjoint) {
        int64_t id;
        std::vector<int32_t> stateModes(wires.size());
        std::transform(wires.begin(), wires.end(), stateModes.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyTensorOperator(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* int32_t numStateModes */ stateModes.size(),
            /* const int32_t * stateModes */ stateModes.data(),
            /* void * */ static_cast<void *>(gateTensorPtr),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 1,
            /* const int32_t adjoint */ adjoint,
            /* const int32_t unitary */ 1,
            /* int64_t * */ &id));
    }

    void applyControlledGate_(CFP_t *gateTensorPtr,
                              const std::vector<size_t> &ctrls,
                              const std::vector<size_t> &tgts, bool adjoint) {

        int64_t id;

        std::vector<int32_t> stateControlModes(ctrls.size());
        std::vector<int32_t> stateTargetModes(ctrls.size());

        std::transform(ctrls.begin(), ctrls.end(), stateControlModes.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        std::transform(tgts.begin(), tgts.end(), stateTargetModes.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyControlledTensorOperator(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* int32_t numControlModes*/ static_cast<int32_t>(ctrls.size()),
            /* const int32_t *stateControlModes*/ stateControlModes.data(),
            /* const int64_t *stateControlValues*/ nullptr,
            /* int32_t numTargetModes*/ static_cast<int32_t>(tgts.size()),
            /* const int32_t *stateTargetModes*/ stateTargetModes.data(),
            /* void *tensorData */ static_cast<void *>(gateTensorPtr),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 1,
            /* const int32_t adjoint */ adjoint,
            /* const int32_t unitary*/ 1,
            /* int64_t *tensorId*/ &id));
    }

    // ComplexT expval_(CFP_t *gateTensorPtr, const std::vector<size_t> &wires)
    // {
    ComplexT expval_(cutensornetNetworkOperator_t tnOps) {
        ComplexT expectVal{0.0, 0.0}, stateNorm2{0.0, 0.0};

        cutensornetStateExpectation_t expectation;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateExpectation(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetNetworkOperator_t */ tnOps,
            /* cutensornetStateExpectation_t * */ &expectation));

        // Configure the computation of the specified quantum circuit
        // expectation value
        const int32_t numHyperSamples =
            8; // desired number of hyper samples used in the tensor network
               // contraction path finder

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationConfigure(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetExpectationAttributes_t */
            CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES,
            /* const void * */ &numHyperSamples,
            /* size_t */ sizeof(numHyperSamples)));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetWorkspaceDescriptor_t * */ &workDesc));

        const std::size_t scratchSize = getScratchMemorySize();

        // Prepare the specified quantum circuit expectation value for
        // computation
        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationPrepare(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t [unused] */ 0x0));

        int64_t worksize = this->getWorkSpaceMemorySize_(workDesc);

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device.\n");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<size_t, int> d_scratch(d_scratch_length,
                                          BaseType::getDevTag(), true);

        this->setWorkSpaceMemory_(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationCompute(
            /* const cutensornetHandle_t */ BaseType::getCutnHandle(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* void* */ static_cast<void *>(&expectVal),
            /* void* */ static_cast<void *>(&stateNorm2),
            /*  cudaStream_t unused */ 0x0));

        expectVal /= stateNorm2;

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyExpectation(expectation));

        return expectVal;
    }
};
} // namespace Pennylane::LightningTensor