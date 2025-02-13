// Copyright 2024 Xanadu Quantum Technologies Inc.

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
 * @file TNCuda.hpp
 * Base class for cuTensorNet-backed tensor networks (for common APIs
 * of MPS and ExactTN).
 */

#pragma once

#include <complex>
#include <memory>
#include <set>
#include <vector>

#include <cuda.h>

#include "LinearAlg.hpp"
#include "TNCudaBase.hpp"
#include "TNCudaGateCache.hpp"
#include "TensorBase.hpp"
#include "TensorCuda.hpp"

#include "Util.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Gates;
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
///@endcond

namespace Pennylane::LightningTensor::TNCuda {
/**
 * @brief CRTP-enabled base class for cuTensorNet backends. This class stores
 * both tensor data and gate data supported by both ExactTN and MPS.
 *
 * @tparam PrecisionT Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */
template <class PrecisionT, class Derived>
class TNCuda : public TNCudaBase<PrecisionT, Derived> {
  private:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using ComplexT = std::complex<PrecisionT>;
    using BaseType = TNCudaBase<PrecisionT, Derived>;

  public:
    TNCuda() = delete;

    explicit TNCuda(std::size_t numQubits, std::size_t maxBondDim = 1)
        : BaseType(numQubits), maxBondDim_(maxBondDim),
          bondDims_(setBondDims_()), sitesModes_(setSitesModes_()),
          sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()),
          cublascaller_(make_shared_cublas_caller()),
          gate_cache_(std::make_shared<TNCudaGateCache<PrecisionT>>(
              BaseType::getDevTag())) {
        initTensors_();
        appendInitialMPSState_(
            getSitesExtentsPtr()
                .data()); // This API works for Exact Tensor Network as well,
                          // given sitesExtents_ settings meet the requirement
                          // of cutensornet.
    }

    explicit TNCuda(const std::size_t numQubits, DevTag<int> dev_tag,
                    const std::size_t maxBondDim = 1)
        : BaseType(numQubits, dev_tag.getDeviceID(), dev_tag.getStreamID()),
          maxBondDim_(maxBondDim), bondDims_(setBondDims_()),
          sitesModes_(setSitesModes_()), sitesExtents_(setSitesExtents_()),
          sitesExtents_int64_(setSitesExtents_int64_()),
          cublascaller_(make_shared_cublas_caller()),
          gate_cache_(std::make_shared<TNCudaGateCache<PrecisionT>>(
              BaseType::getDevTag())) {
        initTensors_();
        appendInitialMPSState_(
            getSitesExtentsPtr()
                .data()); // This API works for Exact Tensor Network as well,
                          // given sitesExtents_ settings meet the requirement
                          // of cutensornet.
    }

    ~TNCuda() = default;

    /**
     * @brief Get the method of a derived class object.
     *
     * @return std::string
     */
    [[nodiscard]] auto getMethod() const -> std::string {
        return Derived::method;
    }

    /**
     * @brief Access the CublasCaller the object is using.
     *
     * @return a reference to the object's CublasCaller object.
     */
    auto getCublasCaller() const -> const CublasCaller & {
        return *cublascaller_;
    }

    /**
     * @brief Get the full state vector representation of a quantum state.
     * Note that users/developers should be responsible to ensure that there is
     * sufficient memory on the host to store the full state vector.
     *
     * @param res Pointer to the host memory to store the full state vector
     * @param res_length Length of the result vector
     */
    void getData(ComplexT *res, const std::size_t res_length) {
        PL_ABORT_IF_NOT(res_length ==
                            Pennylane::Util::exp2(BaseType::getNumQubits()),
                        "The size of the result vector should be equal to the "
                        "dimension of the quantum state.");

        std::size_t avail_gpu_memory = getFreeMemorySize();

        PL_ABORT_IF(avail_gpu_memory <
                        Pennylane::Util::exp2(BaseType::getNumQubits()) *
                            sizeof(ComplexT),
                    "State tensor size exceeds the available GPU memory!");
        get_state_tensor(res);
    }

    /**
     * @brief Get the full state vector representation of a quantum state.
     *
     * @return std::vector<ComplexT> Full state vector representation of the
     * quantum state on host
     */
    auto getDataVector() -> std::vector<ComplexT> {
        std::size_t length = Pennylane::Util::exp2(BaseType::getNumQubits());
        std::vector<ComplexT> results(length);

        getData(results.data(), results.size());

        return results;
    }

    /**
     * @brief Set current quantum state as zero state.
     */
    void reset() {
        const std::vector<std::size_t> zeroState(BaseType::getNumQubits(), 0);
        setBasisState(zeroState);
    }

    /**
     * @brief Update quantum state with a basis state.
     * NOTE: This API assumes the bond vector is a standard basis vector
     * ([1,0,0,......]) and current implementation only works for qubit systems.
     * @param basisState Vector representation of a basis state.
     */
    void setBasisState(const std::vector<std::size_t> &basisState) {
        PL_ABORT_IF(BaseType::getNumQubits() != basisState.size(),
                    "The size of a basis state should be equal to the number "
                    "of qubits.");

        bool allZeroOrOne = std::all_of(
            basisState.begin(), basisState.end(),
            [](std::size_t bitVal) { return bitVal == 0 || bitVal == 1; });

        PL_ABORT_IF_NOT(allZeroOrOne,
                        "Please ensure all elements of a basis state should be "
                        "either 0 or 1.");

        CFP_t value_cu = cuUtil::complexToCu<ComplexT>(ComplexT{1.0, 0.0});

        tensors_[0].getDataBuffer().zeroInit();
        std::size_t idx = BaseType::getNumQubits() - std::size_t{1};
        std::size_t target = basisState[idx];
        PL_CUDA_IS_SUCCESS(
            cudaMemcpy(&tensors_[0].getDataBuffer().getData()[target],
                       &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));

        for (std::size_t i = 1; i < BaseType::getNumQubits(); i++) {
            tensors_[i].getDataBuffer().zeroInit();

            idx = BaseType::getNumQubits() - std::size_t{1} - i;
            target = basisState[idx] == 0 ? 0 : bondDims_[i - 1];

            PL_CUDA_IS_SUCCESS(
                cudaMemcpy(&tensors_[i].getDataBuffer().getData()[target],
                           &value_cu, sizeof(CFP_t), cudaMemcpyHostToDevice));
        }
    };

    /**
     * @brief Update the ith tensor data.
     *
     * @param site_idx Index of the site.
     * @param host_data Pointer to the data on host.
     * @param host_data_size Length of the data.
     */
    void updateSiteData(const std::size_t site_idx, const ComplexT *host_data,
                        std::size_t host_data_size) {
        PL_ABORT_IF_NOT(
            site_idx < BaseType::getNumQubits(),
            "The site index should be less than the number of qubits.");

        const std::size_t idx = BaseType::getNumQubits() - site_idx - 1;
        PL_ABORT_IF_NOT(
            host_data_size == tensors_[idx].getDataBuffer().getLength(),
            "The length of the host data should match its copy on the device.");

        tensors_[idx].getDataBuffer().zeroInit();

        tensors_[idx].getDataBuffer().CopyHostDataToGpu(host_data,
                                                        host_data_size);
    }

    /**
     * @brief Append multiple gates to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     * @param ops_params Vector of gate parameters.
     */
    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<std::size_t>> &ops_wires,
                    const std::vector<bool> &ops_adjoint,
                    const std::vector<std::vector<PrecisionT>> &ops_params) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_adjoint[i], ops_params[i]);
        }
    }

    /**
     * @brief Append multiple gate tensors to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<std::size_t>> &ops_wires,
                         const std::vector<bool> &ops_adjoint) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            applyOperation(ops[i], ops_wires[i], ops_adjoint[i], {});
        }
    }

    /**
     * @brief Append a single controlled gate tensor to the compute graph.
     *
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph. The controlled gate should be
     * immutable as v24.08.
     *
     * @param baseOpName Base gate's name.
     * @param controlled_wires Controlled wires for the gate.
     * @param controlled_values Controlled values for the gate.
     * @param targetWires Target wires for the gate.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix for custom gates.
     */
    void
    applyControlledOperation(const std::string &baseOpName,
                             const std::vector<std::size_t> &controlled_wires,
                             const std::vector<bool> &controlled_values,
                             const std::vector<std::size_t> &targetWires,
                             bool adjoint = false,
                             const std::vector<PrecisionT> &params = {0.0},
                             const std::vector<ComplexT> &gate_matrix = {}) {
        // TODO: Need to revisit this line of code once `cutensornet` supports
        // multi-target wire controlled gates
        PL_ABORT_IF_NOT(targetWires.size() == 1,
                        "Unsupported controlled gate: cutensornet only "
                        "supports 1-wire target controlled gates");

        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        int64_t dummy_id = gate_ids_.empty() ? 1 : *gate_ids_.rbegin() + 1;

        if (gate_matrix.empty()) {
            gate_cache_->add_gate(dummy_id, baseOpName, par, adjoint);
        } else {
            auto gate_key = std::make_pair(baseOpName, par);
            std::vector<CFP_t> matrix_cu =
                cuUtil::complexToCu<ComplexT>(gate_matrix);
            gate_cache_->add_gate(dummy_id, gate_key, matrix_cu, adjoint);
        }

        int64_t id;

        std::vector<int32_t> controlledModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                controlled_wires, BaseType::getNumQubits());

        std::vector<int64_t> controlled_values_int64(controlled_values.size());
        std::transform(controlled_values.begin(), controlled_values.end(),
                       controlled_values_int64.begin(),
                       [](bool val) { return static_cast<int64_t>(val); });

        std::vector<int32_t> targetModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                targetWires, BaseType::getNumQubits());

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyControlledTensorOperator(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* int32_t numControlModes */ controlled_wires.size(),
            /* const int32_t * stateControlModes */ controlledModes.data(),
            /* const int64_t *stateControlValues*/
            controlled_values_int64.data(),
            /* int32_t numTargetModes */ targetWires.size(),
            /* const int32_t * stateTargetModes */ targetModes.data(),
            /* void * */
            static_cast<void *>(gate_cache_->get_gate_device_ptr(dummy_id)),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 1,
            /* const int32_t adjoint */ 0,
            /* const int32_t unitary */ 1,
            /* int64_t tensorId* */ &id));

        if (dummy_id != id) {
            gate_cache_->update_key(dummy_id, id);
        }

        gate_ids_.insert(id);
    }

    /**
     * @brief Append a single gate tensor to the compute graph.
     * NOTE: This function does not update the quantum state but only appends
     * gate tensor operator to the graph.
     * @param opName Gate's name.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional gate matrix for custom gates.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool adjoint = false,
                        const std::vector<PrecisionT> &params = {0.0},
                        const std::vector<ComplexT> &gate_matrix = {}) {
        PL_ABORT_IF(
            wires.size() > 2 && getMethod() == "mps",
            "Unsupported gate: MPS method only supports 1, 2-wires gates");

        auto &&par = (params.empty()) ? std::vector<PrecisionT>{0.0} : params;

        int64_t dummy_id = gate_ids_.empty() ? 1 : *gate_ids_.rbegin() + 1;

        if (gate_matrix.empty()) [[likely]] {
            gate_cache_->add_gate(dummy_id, opName, par, adjoint);
        } else [[unlikely]] {
            auto gate_key = std::make_pair(opName, par);
            std::vector<CFP_t> matrix_cu =
                cuUtil::complexToCu<ComplexT>(gate_matrix);
            gate_cache_->add_gate(dummy_id, gate_key, matrix_cu, adjoint);
        }

        int64_t id;

        std::vector<int32_t> stateModes =
            cuUtil::NormalizeCastIndices<std::size_t, int32_t>(
                wires, BaseType::getNumQubits());

        //  Note `adjoint` in the cutensornet context indicates whether or not
        //  all tensor elements of the tensor operator will be complex
        //  conjugated. `adjoint` in the following API is not equivalent to
        //  `inverse` in the lightning context
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateApplyTensorOperator(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* int32_t numStateModes */ stateModes.size(),
            /* const int32_t * stateModes */ stateModes.data(),
            /* void * */
            static_cast<void *>(gate_cache_->get_gate_device_ptr(dummy_id)),
            /* const int64_t *tensorModeStrides */ nullptr,
            /* const int32_t immutable */ 0,
            /* const int32_t adjoint */ 0,
            /* const int32_t unitary */ 1,
            /* int64_t * */ &id));

        if (dummy_id != id) {
            gate_cache_->update_key(dummy_id, id);
        }

        // one time initialization of the identity gate id
        if (identiy_gate_ids_.empty() && opName == "Identity") {
            identiy_gate_ids_.push_back(static_cast<std::size_t>(id));
        }

        gate_ids_.insert(id);
    }

    /**
     * @brief Get the state vector representation of a tensor network.
     *
     * @param host_data Pointer to the host memory for state tensor data.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_state_tensor(ComplexT *host_data,
                          const int32_t numHyperSamples = 1) {
        std::vector<int32_t> projected_modes{};
        std::vector<int64_t> projected_mode_values{};

        const std::size_t length = std::size_t{1} << BaseType::getNumQubits();

        DataBuffer<CFP_t, int> d_output_tensor(length, BaseType::getDevTag(),
                                               true);

        get_accessor_(d_output_tensor.getData(), length, projected_modes,
                      projected_mode_values, numHyperSamples);

        d_output_tensor.CopyGpuDataToHost(host_data, length);
    }

    /**
     * @brief Get a slice of the full state tensor.
     *
     * @param tensor_data Pointer to the device memory for state tensor data.
     * @param tensor_data_size Size of the state tensor data.
     * @param projected_modes Projected modes to get the state tensor for.
     * @param projected_mode_values Values of the projected modes.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_state_tensor(CFP_t *tensor_data,
                          const std::size_t tensor_data_size,
                          const std::vector<int32_t> &projected_modes,
                          const std::vector<int64_t> &projected_mode_values,
                          const int32_t numHyperSamples = 1) const {
        get_accessor_(tensor_data, tensor_data_size, projected_modes,
                      projected_mode_values, numHyperSamples);
    }

    /**
     * @brief Get a const vector reference of sitesExtents_.
     *
     * @return const std::vector<std::vector<std::size_t>>
     */
    [[nodiscard]] auto getSitesExtents() const
        -> const std::vector<std::vector<std::size_t>> & {
        return sitesExtents_;
    }

  protected:
    // Note both maxBondDim_ and bondDims_ are used for both MPS and Exact
    // Tensor Network. For Exact Tensor Network, maxBondDim_ is 1 and bondDims_
    // is {1}. For Exact Tensor Network, setting bondDims_ allows call to
    // appendInitialMPSState_() to append the initial state to the Exact Tensor
    // Network state.
    const std::size_t
        maxBondDim_; // maxBondDim_ default is 1 for Exact Tensor Network
    const std::vector<std::size_t>
        bondDims_; // bondDims_ default is {1} for Exact Tensor Network

    /**
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<CFP_t *>
     */
    [[nodiscard]] auto getTensorsOutDataPtr() -> std::vector<CFP_t *> {
        std::vector<CFP_t *> tensorsOutDataPtr(BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensorsOutDataPtr[i] = tensors_out_[i].getDataBuffer().getData();
        }
        return tensorsOutDataPtr;
    }

    /**
     * @brief Get a vector of pointers to extents of each site.
     *
     * @return std::vector<int64_t const *> Note int64_t const* is
     * required by cutensornet backend.
     */
    [[nodiscard]] auto getSitesExtentsPtr() -> std::vector<int64_t const *> {
        std::vector<int64_t const *> sitesExtentsPtr_int64(
            BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            sitesExtentsPtr_int64[i] = sitesExtents_int64_[i].data();
        }
        return sitesExtentsPtr_int64;
    }

    /**
     * @brief Save quantumState information to data provided by a user
     *
     * @param extentsPtr Pointer to extents provided by a user
     * @param tensorPtr Pointer to tensors provided by a user
     */
    void computeState(int64_t **extentsPtr, void **tensorPtr) {
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            BaseType::getTNCudaHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStatePrepare(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* std::size_t maxWorkspaceSizeDevice */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /*  cudaStream_t unused as of v24.03*/ 0x0));

        std::size_t worksize =
            getWorkSpaceMemorySize(BaseType::getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t);
        DataBuffer<std::size_t, int> d_scratch(d_scratch_length,
                                               BaseType::getDevTag(), true);

        setWorkSpaceMemory(BaseType::getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateCompute(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* int64_t * */ extentsPtr,
            /* int64_t *stridesOut */ nullptr,
            /* void * */ tensorPtr,
            /* cudaStream_t */ BaseType::getDevTag().getStreamID()));

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
    }

  private:
    const std::vector<std::vector<std::size_t>> sitesModes_;
    const std::vector<std::vector<std::size_t>> sitesExtents_;
    const std::vector<std::vector<int64_t>> sitesExtents_int64_;

    SharedCublasCaller cublascaller_;

    std::shared_ptr<TNCudaGateCache<PrecisionT>> gate_cache_;
    std::set<int64_t> gate_ids_;

    std::vector<std::size_t> identiy_gate_ids_;

    std::vector<TensorCuda<PrecisionT>> tensors_;
    std::vector<TensorCuda<PrecisionT>> tensors_out_;

    /**
     * @brief Get accessor of a state tensor
     *
     * @param tensor_data Pointer to the device memory for state tensor data.
     * @param tensor_data_size Size of the tensor data.
     * @param projected_modes Projected modes to get the state tensor for.
     * @param projected_mode_values Values of the projected modes.
     * @param numHyperSamples Number of hyper samples to use in the calculation
     * and is set to 1 by default.
     */
    void get_accessor_(CFP_t *tensor_data,
                       [[maybe_unused]] const std::size_t tensor_data_size,
                       const std::vector<int32_t> &projected_modes,
                       const std::vector<int64_t> &projected_mode_values,
                       const int32_t numHyperSamples = 1) const {
        cutensornetStateAccessor_t accessor;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateAccessor(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetState_t */ BaseType::getQuantumState(),
            /* int32_t numProjectedModes */
            static_cast<int32_t>(projected_modes.size()),
            /* const int32_t *projectedModes */ projected_modes.data(),
            /* const int64_t *amplitudesTensorStrides */ nullptr,
            /* cutensornetStateAccessor_t *tensorNetworkAccessor*/ &accessor));

        // Configure the computation
        const cutensornetAccessorAttributes_t accessor_attribute =
            CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorConfigure(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetStateAccessor_t */ accessor,
            /* cutensornetAccessorAttributes_t */ accessor_attribute,
            /* const void * */ &numHyperSamples,
            /* std::size_t */ sizeof(numHyperSamples)));

        // prepare the computation
        cutensornetWorkspaceDescriptor_t workDesc;
        PL_CUTENSORNET_IS_SUCCESS(cutensornetCreateWorkspaceDescriptor(
            BaseType::getTNCudaHandle(), &workDesc));

        // TODO we assign half (magic number is) of free memory size to the
        // maximum memory usage.
        const std::size_t scratchSize = cuUtil::getFreeMemorySize() / 2;

        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorPrepare(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetStateAccessor_t*/ accessor,
            /* std::size_t */ scratchSize,
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* cudaStream_t unused as of v24.03 */ 0x0));

        // Allocate workspace buffer
        std::size_t worksize =
            getWorkSpaceMemorySize(BaseType::getTNCudaHandle(), workDesc);

        PL_ABORT_IF(worksize > scratchSize,
                    "Insufficient workspace size on Device!");

        const std::size_t d_scratch_length = worksize / sizeof(std::size_t);
        DataBuffer<std::size_t, int> d_scratch(d_scratch_length,
                                               BaseType::getDevTag(), true);

        setWorkSpaceMemory(BaseType::getTNCudaHandle(), workDesc,
                           reinterpret_cast<void *>(d_scratch.getData()),
                           worksize);

        // compute the specified slice of the quantum circuit amplitudes tensor
        ComplexT stateNorm2{0.0, 0.0};
        PL_CUTENSORNET_IS_SUCCESS(cutensornetAccessorCompute(
            /* const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /* cutensornetStateAccessor_t */ accessor,
            /* const int64_t * projectedModeValues */
            projected_mode_values.data(),
            /* cutensornetWorkspaceDescriptor_t */ workDesc,
            /* void *amplitudesTensor*/
            static_cast<void *>(tensor_data),
            /* void *stateNorm */ static_cast<void *>(&stateNorm2),
            /* cudaStream_t cudaStream */ 0x0));

        PL_CUDA_IS_SUCCESS(
            cudaStreamSynchronize(BaseType::getDevTag().getStreamID()));

        const ComplexT scale_scalar = ComplexT{1.0, 0.0} / stateNorm2;

        CFP_t scale_scalar_cu{scale_scalar.real(), scale_scalar.imag()};

        scaleC_CUDA<CFP_t, CFP_t>(
            scale_scalar_cu, tensor_data, tensor_data_size,
            BaseType::getDevTag().getDeviceID(),
            BaseType::getDevTag().getStreamID(), getCublasCaller());

        PL_CUTENSORNET_IS_SUCCESS(
            cutensornetDestroyWorkspaceDescriptor(workDesc));
        PL_CUTENSORNET_IS_SUCCESS(cutensornetDestroyAccessor(accessor));
    }

    /**
     * @brief Append initial MPS sites to the compute graph with data provided
     * by a user
     *
     */
    void appendInitialMPSState_(const int64_t *const *extentsPtr) {
        PL_CUTENSORNET_IS_SUCCESS(cutensornetStateInitializeMPS(
            /*const cutensornetHandle_t */ BaseType::getTNCudaHandle(),
            /*cutensornetState_t*/ BaseType::getQuantumState(),
            /*cutensornetBoundaryCondition_t */
            CUTENSORNET_BOUNDARY_CONDITION_OPEN,
            /*const int64_t *const* */ extentsPtr,
            /*const int64_t *const* */ nullptr,
            /*void ** */
            reinterpret_cast<void **>(getTensorsDataPtr_().data())));
    }

    /**
     * @brief Get a vector of pointers to tensor data of each site.
     *
     * @return std::vector<uint64_t *>
     */
    [[nodiscard]] auto getTensorsDataPtr_() -> std::vector<uint64_t *> {
        std::vector<uint64_t *> tensorsDataPtr(BaseType::getNumQubits());
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            tensorsDataPtr[i] = reinterpret_cast<uint64_t *>(
                tensors_[i].getDataBuffer().getData());
        }
        return tensorsDataPtr;
    }

    /**
     * @brief Return bondDims to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::size_t>
     */
    std::vector<std::size_t> setBondDims_() {
        std::vector<std::size_t> localBondDims(BaseType::getNumQubits() - 1,
                                               maxBondDim_);

        const std::size_t ubDim = log2(maxBondDim_);
        for (std::size_t i = 0; i < localBondDims.size(); i++) {
            const std::size_t bondDim =
                std::min(i + 1, BaseType::getNumQubits() - i - 1);

            if (bondDim <= ubDim) {
                localBondDims[i] = std::size_t{1} << bondDim;
            }
        }
        return localBondDims;
    }

    /**
     * @brief Return siteModes to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesModes_() {
        std::vector<std::vector<std::size_t>> localSitesModes;
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteModes;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteModes =
                    std::vector<std::size_t>({i, i + BaseType::getNumQubits()});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteModes = std::vector<std::size_t>(
                    {i + BaseType::getNumQubits() - 1, i});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteModes =
                    std::vector<std::size_t>({i + BaseType::getNumQubits() - 1,
                                              i, i + BaseType::getNumQubits()});
            }
            localSitesModes.push_back(std::move(localSiteModes));
        }
        return localSitesModes;
    }

    /**
     * @brief Return sitesExtents to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<std::size_t>>
     */
    std::vector<std::vector<std::size_t>> setSitesExtents_() {
        std::vector<std::vector<std::size_t>> localSitesExtents;

        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            std::vector<std::size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {BaseType::getQubitDims()[i], bondDims_[i]});
            } else if (i == BaseType::getNumQubits() - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims_[i - 1], BaseType::getQubitDims()[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteExtents = std::vector<std::size_t>(
                    {bondDims_[i - 1], BaseType::getQubitDims()[i],
                     bondDims_[i]});
            }
            localSitesExtents.push_back(std::move(localSiteExtents));
        }
        return localSitesExtents;
    }

    /**
     * @brief Return siteExtents_int64 to the member initializer
     * NOTE: This method only works for the open boundary condition
     * @return std::vector<std::vector<int64_t>>
     */
    std::vector<std::vector<int64_t>> setSitesExtents_int64_() {
        std::vector<std::vector<int64_t>> localSitesExtents_int64;

        for (const auto &siteExtents : sitesExtents_) {
            localSitesExtents_int64.push_back(
                std::move(Pennylane::Util::cast_vector<std::size_t, int64_t>(
                    siteExtents)));
        }
        return localSitesExtents_int64;
    }

    /**
     * @brief The tensors init helper function for ctor.
     */
    void initTensors_() {
        for (std::size_t i = 0; i < BaseType::getNumQubits(); i++) {
            // construct mps tensors reprensentation
            tensors_.emplace_back(sitesModes_[i].size(), sitesModes_[i],
                                  sitesExtents_[i], BaseType::getDevTag());

            tensors_out_.emplace_back(sitesModes_[i].size(), sitesModes_[i],
                                      sitesExtents_[i], BaseType::getDevTag());
        }
        reset();
    }
};
} // namespace Pennylane::LightningTensor::TNCuda
