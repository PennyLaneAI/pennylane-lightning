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
 * @file MPS_cuDevice.hpp
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
#include "TensorBase.hpp"
#include "cuDeviceTensor.hpp"
#include "cuGateTensorCache.hpp"
#include "cuTensorNetError.hpp"
#include "cuTensorNet_helpers.hpp"
#include "cuda_helpers.hpp"

namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::Util;

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

namespace Pennylane::LightningTensor {

template <class Precision> class cuMPS {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(Precision{}));
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;

  private:
    SharedCutnHandle handle_;
    cudaDataType_t typeData_;
    cutensornetComputeType_t typeCompute_;
    cutensornetState_t quantumState_;
    cutensornetStatePurity_t purity_ =
        CUTENSORNET_STATE_PURITY_PURE; // Only supports pure tensor network
                                       // states as v24.03

    size_t numQubits_;
    size_t maxExtent_;
    std::vector<size_t> qubitDims_;

    Pennylane::LightningGPU::DevTag<int> dev_tag_;

    std::vector<cuDeviceTensor<Precision>> d_mpsTensors_;

    std::shared_ptr<GateTensorCache<Precision>> gate_cache_;

  public:
    // TODO add SVD options by the cutensornetStateConfigure() API
    //  cutensornetStateAttributes_t,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF,
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_S_NORMALIZATION
    //  CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO
    cuMPS(size_t &numQubits, size_t &maxExtent, std::vector<size_t> &qubitDims,
          Pennylane::LightningGPU::DevTag<int> &dev_tag)
        : handle_(make_shared_cutn_handle()), numQubits_(numQubits),
          maxExtent_(maxExtent), qubitDims_(qubitDims), dev_tag_(dev_tag),
          gate_cache_(std::make_shared<GateTensorCache<Precision>>(true, dev_tag)) {

        if constexpr (std::is_same_v<Precision, double>) {
            typeData_ = CUDA_C_64F;
            typeCompute_ = CUTENSORNET_COMPUTE_64F;
        } else {
            typeData_ = CUDA_C_32F;
            typeCompute_ = CUTENSORNET_COMPUTE_32F;
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateState(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStatePurity_t */ purity_,
            /* int32_t numStateModes */ static_cast<int32_t>(numQubits_),
            /* const int64_t *stateModeExtents */
            reinterpret_cast<int64_t *>(qubitDims_.data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetState_t * */ &quantumState_));

        for (size_t i = 0; i < numQubits_; i++) {
            // size_t siteRank;
            std::vector<size_t> modes;
            std::vector<size_t> siteExtents;
            if (i == 0) {
                // L
                modes = std::vector<size_t>({i, i + numQubits_});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else if (i == numQubits_ - 1) {
                // R
                modes = std::vector<size_t>({i + numQubits_, i});
                siteExtents = std::vector<size_t>({qubitDims[i], maxExtent_});
            } else {
                // M
                modes = std::vector<size_t>(
                    {i + numQubits_ - 1, i, i + numQubits_});
                siteExtents =
                    std::vector<size_t>({maxExtent_, qubitDims[i], maxExtent_});
            }
            d_mpsTensors_.emplace_back(modes.size(), modes, siteExtents,
                                       dev_tag_);
        }
    }

    ~cuMPS() {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyState(quantumState_));
    }

    auto getDevTag() const -> Pennylane::LightningGPU::DevTag<int> {
        return dev_tag_;
    }

    auto getGateCache() -> GateTensorCache<Precision> * {
        return gate_cache_.get();
    }

    /**
     * @brief Get the cuTensorNet handle that the object is using.
     *
     * @return cutensornetHandle_t returns the cuTensorNet handle.
     */
    auto getCutnHandle() const -> cutensornetHandle_t { return handle_.get(); }

    // Set a zero state for d_mpsTensors
    void reset() {
        size_t index = 0;
        this->setBasisState(index);
    }

    void setBasisState(size_t index) {
        // Assuming the site vector is [1,0] or [0,1] and bond vector is
        // [1,0,0...].
        std::string str = size_t_to_binary_string(numQubits_, index);

        std::cout << str << std::endl;

        CFP_t value_cu = Pennylane::LightningGPU::Util::complexToCu<
            std::complex<Precision>>({1.0, 0.0});

        for (size_t i = 0; i < d_mpsTensors_.size(); i++) {
            d_mpsTensors_[i].getDataBuffer().zeroInit();

            size_t target = 0;

            if (i == 0) {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : 1;
            } else if (i == numQubits_ - 1) {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : maxExtent_;
            } else {
                target = str.at(numQubits_ - 1 - i) == '0' ? 0 : maxExtent_;
            }

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&d_mpsTensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }

        std::vector<std::vector<int64_t>> extents;
        std::vector<int64_t *> extentsPtr(numQubits_);
        std::vector<void *> mpsTensorsDataPtr(numQubits_, nullptr);

        for (size_t i = 0; i < numQubits_; i++) {
            std::vector<int64_t> localExtents(
                d_mpsTensors_[i].getExtents().size());

            for (size_t j = 0; j < d_mpsTensors_[i].getExtents().size(); j++) {
                localExtents[j] =
                    static_cast<int64_t>(d_mpsTensors_[i].getExtents()[j]);
            }

            extents.push_back(localExtents);

            extentsPtr[i] = extents[i].data();
            mpsTensorsDataPtr[i] =
                static_cast<void *>(d_mpsTensors_[i].getDataBuffer().getData());
        }

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t*/ handle_.get(),
            /*cutensornetState_t*/ quantumState_,
            /*cutensornetBoundaryCondition_t*/
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const*/ extentsPtr.data(),
            /*const int64_t *const*/ nullptr,
            /*void **/ mpsTensorsDataPtr.data()));
    };

    auto getStateVector() -> std::vector<std::complex<Precision>> {
        // 1D representation of mpsTensor
        std::vector<size_t> modes(1, 1);
        std::vector<size_t> extent(1, (1 << numQubits_));
        cuDeviceTensor<Precision> d_mpsTensor(modes.size(), modes, extent,
                                               dev_tag_);

        std::vector<void *> d_mpsTensorsPtr(
            1, static_cast<void *>(d_mpsTensor.getDataBuffer().getData()));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetCreateWorkspaceDescriptor(handle_.get(), &workDesc));

        const std::size_t scratchSize = getScratchMemorySize();

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetState_t */ quantumState_,
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused in v24.03*/ 0x0));

        int64_t worksize = this->getWorkSpaceMemorySize_(workDesc);

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<size_t, int> d_scratch(d_scratch_length, dev_tag_, true);

        this->setWorkSpaceMemory_(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        std::vector<int64_t *> extentsPtr;
        std::vector<int64_t> extent_int64(1, (1 << numQubits_));
        extentsPtr.emplace_back(extent_int64.data());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetState_t */ quantumState_,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr.data(),
            /* int64_t *stridesOut */ nullptr,
            /* void * */ d_mpsTensorsPtr.data(),
            /* cudaStream_t */ dev_tag_.getStreamID()));

        std::vector<ComplexT> results(extent.front());

        d_mpsTensor.CopyGpuDataToHost(results.data(), results.size());

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        return results;
    }

    void applyOperation(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false,
                        const std::vector<Precision> &params = {0.0}) {
        PL_ABORT_IF(wires.size() > 2,
                    "Current version only supports 1/2 qubit gates.");
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        applyGate_(gate_cache_->get_gate_device_ptr(opName, par[0]), wires,
                   adjoint);
    }

    /**
     * @brief Return the mapping of named gates to amount of control wires they
     * have.
     *
     * @return const std::unordered_map<std::string, std::size_t>&
     */
    auto getCtrlMap() -> const std::unordered_map<std::string, std::size_t> & {
        return ctrl_map_;
    }

    void applyGeneralOperation(const std::string &opName,
                               const std::vector<size_t> &wires,
                               bool adjoint = false,
                               const std::vector<Precision> &params = {0.0}) {
        PL_ABORT_IF(wires.size() > 2,
                    "Current version only supports 1/2 qubit gates.");

        const auto ctrl_offset =
            (this->getCtrlMap().find(opName) != this->getCtrlMap().end())
                ? this->getCtrlMap().at(opName)
                : 0;
        const std::vector<std::size_t> ctrls{wires.begin(),
                                             wires.begin() + ctrl_offset};
        const std::vector<std::size_t> tgts{wires.begin() + ctrl_offset,
                                            wires.end()};

        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;

        if (ctrls.size() > 0) {
            applyControlledGate_(
                gate_cache_->get_gate_device_ptr(opName, par[0]), ctrls, tgts,
                adjoint);
        } else {
            applyGate_(gate_cache_->get_gate_device_ptr(opName, par[0]), wires,
                       adjoint);
        }
    }

    ComplexT expval(const std::string &opName, const std::vector<size_t> &wires,
                    const std::vector<Precision> &params = {0.0}) {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        return expval_(gate_cache_->get_gate_device_ptr(opName, par[0]), wires);
    }

  private:
    size_t getWorkSpaceMemorySize_(cutensornetWorkspaceDescriptor_t &workDesc) {
        int64_t worksize{0};

        PL_CUTENSORNET_IS_SUCCESS(cutensornetWorkspaceGetMemorySize(
            /* const cutensornetHandle_t */ handle_.get(),
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
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cutensornetMemspace_t*/ CUTENSORNET_MEMSPACE_DEVICE,
            /* cutensornetWorkspaceKind_t */ CUTENSORNET_WORKSPACE_SCRATCH,
            /* void *const */ scratchPtr,
            /* int64_t */ worksize));
    }

    void applyGate_(CFP_t *gateTensorPtr, const std::vector<size_t> &wires,
                    bool adjoint) {
        int64_t id;
        std::vector<int32_t> stateModes(wires.size());
        std::transform(wires.begin(), wires.end(), stateModes.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyTensorOperator(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetState_t */ quantumState_,
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
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetState_t */ quantumState_,
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

    ComplexT expval_(CFP_t *gateTensorPtr, const std::vector<size_t> &wires) {

        // Compute the specified quantum circuit expectation value
        ComplexT expectVal{0.0, 0.0}, stateNorm2{0.0, 0.0};

        // TODO add create-tensor-network-operator to observable_cuMPS classes.
        // TODO cutensornetNetworkOperator_t tnOps as private data of
        // observable_cuMPS classes.
        // TODO tnOps can be created with obs->create-tensor-network-operator()
        // method in the Measurement_cuMPS class.
        // TODO move this method to the Measurement_cuMPS class
        // Create an empty tensor network operator
        cutensornetNetworkOperator_t hamiltonian;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateNetworkOperator(
            /* const cutensornetHandle_t */ handle_.get(),
            /* int32_t */ static_cast<int32_t>(numQubits_),
            /* const int64_t stateModeExtents */
            reinterpret_cast<int64_t *>(qubitDims_.data()),
            /* cudaDataType_t */ typeData_,
            /*  cutensornetNetworkOperator_t */ &hamiltonian));

        int64_t id;

        std::vector<int32_t> wires_int(wires.size());
        std::transform(wires.begin(), wires.end(), wires_int.begin(),
                       [](size_t x) { return static_cast<int32_t>(x); });

        std::vector<int32_t> numStateModes(1, wires.size());
        std::vector<const int32_t *> stateModes;
        stateModes.emplace_back(wires_int.data());
        std::vector<const void *> tensorData;
        tensorData.emplace_back(static_cast<const void *>(gateTensorPtr));

        PL_CUTENSORNET_IS_SUCCESS(cutensornetNetworkOperatorAppendProduct(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetNetworkOperator_t */ hamiltonian,
            /* cuDoubleComplex coefficient*/ cuDoubleComplex{1, 0.0},
            /* int32_t numTensors */ 1,
            /* const int32_t numStateModes[] */ numStateModes.data(),
            /* const int32_t *stateModes[] */ stateModes.data(),
            /* const int64_t *tensorModeStrides[] */ nullptr,
            /* const void *tensorData[] */ tensorData.data(),
            /* int64_t* */ &id));

        cutensornetStateExpectation_t expectation;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateExpectation(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetState_t */ quantumState_,
            /* cutensornetNetworkOperator_t */ hamiltonian,
            /* cutensornetStateExpectation_t * */ &expectation));

        // Configure the computation of the specified quantum circuit
        // expectation value
        const int32_t numHyperSamples =
            8; // desired number of hyper samples used in the tensor network
               // contraction path finder

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationConfigure(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetExpectationAttributes_t */
            CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES,
            /* const void * */ &numHyperSamples,
            /* size_t */ sizeof(numHyperSamples)));

        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetWorkspaceDescriptor_t * */ &workDesc));

        const std::size_t scratchSize = getScratchMemorySize();

        // Prepare the specified quantum circuit expectation value for
        // computation
        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationPrepare(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStateExpectation_t */ expectation,
            /* size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t [unused] */ 0x0));

        Precision flops = 0.0;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationGetInfo(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetExpectationAttributes_t */
            CUTENSORNET_EXPECTATION_INFO_FLOPS,
            /* void * */ &flops,
            /* size_t attributeSize */ sizeof(flops)));

        PL_ABORT_IF(flops <= 0.0, "Invalid Flop count.\n");

        int64_t worksize = this->getWorkSpaceMemorySize_(workDesc);

        PL_ABORT_IF(static_cast<std::size_t>(worksize) > scratchSize,
                    "Insufficient workspace size on Device.\n");

        const std::size_t d_scratch_length = worksize / sizeof(size_t) + 1;
        DataBuffer<size_t, int> d_scratch(d_scratch_length, dev_tag_, true);

        this->setWorkSpaceMemory_(
            workDesc, reinterpret_cast<void *>(d_scratch.getData()), worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetExpectationCompute(
            /* const cutensornetHandle_t */ handle_.get(),
            /* cutensornetStateExpectation_t */ expectation,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* void* */ static_cast<void *>(&expectVal),
            /* void* */ static_cast<void *>(&stateNorm2),
            /*  cudaStream_t unused */ 0x0));

        expectVal /= stateNorm2;

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyExpectation(expectation));
        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyNetworkOperator(hamiltonian));

        return expectVal;
    }

  private:
    const std::unordered_set<std::string> const_gates_{
        "Identity", "PauliX", "PauliY", "PauliZ", "Hadamard", "T",      "S",
        "CNOT",     "SWAP",   "CY",     "CZ",     "CSWAP",    "Toffoli"};
    const std::unordered_map<std::string, std::size_t> ctrl_map_{
        // Add mapping from function name to required wires.
        {"Identity", 0},
        {"PauliX", 0},
        {"PauliY", 0},
        {"PauliZ", 0},
        {"Hadamard", 0},
        {"T", 0},
        {"S", 0},
        {"RX", 0},
        {"RY", 0},
        {"RZ", 0},
        {"Rot", 0},
        {"PhaseShift", 0},
        {"ControlledPhaseShift", 1},
        {"CNOT", 1},
        {"SWAP", 0},
        {"CY", 1},
        {"CZ", 1},
        {"CRX", 1},
        {"CRY", 1},
        {"CRZ", 1},
        {"CRot", 1},
        {"CSWAP", 1},
        {"Toffoli", 2}};
};
} // namespace Pennylane::LightningTensor