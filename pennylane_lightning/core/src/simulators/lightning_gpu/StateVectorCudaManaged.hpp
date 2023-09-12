// Copyright 2022-2023 Xanadu Quantum Technologies Inc. and contributors.

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
 * @file StateVectorCudaManaged.hpp
 */
#pragma once

#include <random>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuComplex.h> // cuDoubleComplex
#include <cuda.h>
#include <custatevec.h> // custatevecApplyMatrix

#include "BitUtil.hpp" // isPerfectPowerOf2
#include "Constant.hpp"
#include "Error.hpp"
#include "StateVectorCudaBase.hpp"
#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "cuError.hpp"

#include "LinearAlg.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU {
// declarations of external functions (defined in initSV.cu).
extern void setStateVector_CUDA(cuComplex *sv, int &num_indices,
                                cuComplex *value, int *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);
extern void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                                cuDoubleComplex *value, long *indices,
                                size_t thread_per_block,
                                cudaStream_t stream_id);

extern void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);
extern void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                               const size_t index, bool async,
                               cudaStream_t stream_id);

/**
 * @brief Managed memory CUDA state-vector class using custateVec backed
 * gate-calls.
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision=double>
class StateVectorCudaManaged
    : public StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>> {
  private:
    using BaseType = StateVectorCudaBase<Precision, StateVectorCudaManaged>;

  public:
    using PrecisionT = Precision;
    using ComplexT = std::complex<PrecisionT>;
    using CFP_t =
        typename StateVectorCudaBase<Precision,
                                     StateVectorCudaManaged<Precision>>::CFP_t;
    using GateType = CFP_t *;

    StateVectorCudaManaged() = delete;
    StateVectorCudaManaged(size_t num_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()), gate_cache_(true){};

    StateVectorCudaManaged(
        size_t num_qubits, const DevTag<int> &dev_tag, bool alloc = true,
        SharedCusvHandle cusvhandle_in = make_shared_cusv_handle(),
        SharedCublasCaller cublascaller_in = make_shared_cublas_caller(),
        SharedCusparseHandle cusparsehandle_in = make_shared_cusparse_handle())
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits, dev_tag, alloc),
          handle_(std::move(cusvhandle_in)),
          cublascaller_(std::move(cublascaller_in)),
          cusparsehandle_(std::move(cusparsehandle_in)),
          gate_cache_(true, dev_tag) {
        BaseType::initSV();
    };

    StateVectorCudaManaged(const CFP_t *gpu_data, size_t length)
        : StateVectorCudaManaged(Pennylane::Util::log2(length)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }

    StateVectorCudaManaged(
        const CFP_t *gpu_data, size_t length, DevTag<int> dev_tag,
        SharedCusvHandle handle_in = make_shared_cusv_handle(),
        SharedCublasCaller cublascaller_in = make_shared_cublas_caller(),
        SharedCusparseHandle cusparsehandle_in = make_shared_cusparse_handle())
        : StateVectorCudaManaged(Pennylane::Util::log2(length), dev_tag, true,
                                 std::move(handle_in),
                                 std::move(cublascaller_in),
                                 std::move(cusparsehandle_in)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }

    StateVectorCudaManaged(const std::complex<Precision> *host_data,
                           size_t length)
        : StateVectorCudaManaged(Pennylane::Util::log2(length)) {
        BaseType::CopyHostDataToGpu(host_data, length, false);
    }

    StateVectorCudaManaged(std::complex<Precision> *host_data, size_t length)
        : StateVectorCudaManaged(Pennylane::Util::log2(length)) {
        BaseType::CopyHostDataToGpu(host_data, length, false);
    }

    StateVectorCudaManaged(const StateVectorCudaManaged &other)
        : BaseType(other.getNumQubits(), other.getDataBuffer().getDevTag()),
          handle_(other.handle_), cublascaller_(other.cublascaller_),
          cusparsehandle_(other.cusparsehandle_),
          gate_cache_(true, other.getDataBuffer().getDevTag()) {
        BaseType::CopyGpuDataToGpuIn(other);
    }

    ~StateVectorCudaManaged() = default;

    /**
     * @brief Set value for a single element of the state-vector on device. This
     * method is implemented by cudaMemcpy.
     *
     * @param value Value to be set for the target element.
     * @param index Index of the target element.
     * @param async Use an asynchronous memory copy.
     */
    void setBasisState(const std::complex<Precision> &value, const size_t index,
                       const bool async = false) {
        BaseType::getDataBuffer().zeroInit();

        CFP_t value_cu = cuUtil::complexToCu<std::complex<Precision>>(value);
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();
        setBasisState_CUDA(BaseType::getData(), value_cu, index, async,
                           stream_id);
    }

    /**
     * @brief Set values for a batch of elements of the state-vector. This
     * method is implemented by the customized CUDA kernel defined in the
     * DataBuffer class.
     *
     * @param num_indices Number of elements to be passed to the state vector.
     * @param values Pointer to values to be set for the target elements.
     * @param indices Pointer to indices of the target elements.
     * @param async Use an asynchronous memory copy.
     */
    template <class index_type, size_t thread_per_block = 256>
    void setStateVector(const index_type num_indices,
                        const std::complex<Precision> *values,
                        const index_type *indices, const bool async = false) {
        BaseType::getDataBuffer().zeroInit();

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        index_type num_elements = num_indices;
        DataBuffer<index_type, int> d_indices{
            static_cast<std::size_t>(num_elements), device_id, stream_id, true};
        DataBuffer<CFP_t, int> d_values{static_cast<std::size_t>(num_elements),
                                        device_id, stream_id, true};

        d_indices.CopyHostDataToGpu(indices, d_indices.getLength(), async);
        d_values.CopyHostDataToGpu(values, d_values.getLength(), async);

        setStateVector_CUDA(BaseType::getData(), num_elements,
                            d_values.getData(), d_indices.getData(),
                            thread_per_block, stream_id);
    }

    auto getCustatevecHandle() const -> cusparseHandle_t {
        if (!cusparsehandle_)
            cusparsehandle_ = make_shared_cusparse_handle();
        return cusparsehandle_.get();
    }

    /**
     * @brief Apply a single gate to the state-vector. Offloads to custatevec
     * specific API calls if available. If unable, attempts to use prior cached
     * gate values on the device. Lastly, accepts a host-provided matrix if
     * otherwise, and caches on the device for later reuse.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    void applyOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<CFP_t> &gate_matrix = {}) {
        const auto ctrl_offset = (BaseType::getCtrlMap().find(opName) !=
                                  BaseType::getCtrlMap().end())
                                     ? BaseType::getCtrlMap().at(opName)
                                     : 0;
        const std::vector<std::size_t> ctrls{wires.begin(),
                                             wires.begin() + ctrl_offset};
        const std::vector<std::size_t> tgts{wires.begin() + ctrl_offset,
                                            wires.end()};
        if (opName == "Identity") {
            return;
        } else if (native_gates_.find(opName) != native_gates_.end()) {
            applyParametricPauliGate({opName}, ctrls, tgts, params.front(),
                                     adjoint);
        } else if (opName == "Rot" || opName == "CRot") {
            if (adjoint) {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[2], params[1], params[0]);
                applyHostMatrixGate(rot_matrix, ctrls, tgts, true);
            } else {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[0], params[1], params[2]);
                applyHostMatrixGate(rot_matrix, ctrls, tgts, false);
            }
        } else if (par_gates_.find(opName) != par_gates_.end()) {
            par_gates_.at(opName)(wires, adjoint, params);
        } else { // No offloadable function call; defer to matrix passing
            auto &&par =
                (params.empty()) ? std::vector<Precision>{0.0} : params;
            // ensure wire indexing correctly preserved for tensor-observables
            const std::vector<std::size_t> ctrls_local{ctrls.rbegin(),
                                                       ctrls.rend()};
            const std::vector<std::size_t> tgts_local{tgts.rbegin(),
                                                      tgts.rend()};

            if (!gate_cache_.gateExists(opName, par[0]) &&
                gate_matrix.empty()) {
                std::string message = "Currently unsupported gate: " + opName;
                throw LightningException(message);
            } else if (!gate_cache_.gateExists(opName, par[0])) {
                gate_cache_.add_gate(opName, par[0], gate_matrix);
            }
            applyDeviceMatrixGate(
                gate_cache_.get_gate_device_ptr(opName, par[0]), ctrls_local,
                tgts_local, adjoint);
        }
    }
    /**
     * @brief STL-friendly variant of `applyOperation(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<CFP_t> &gate_matrix = {})`
     *
     */
    void applyOperation_std(
        const std::string &opName, const std::vector<size_t> &wires,
        bool adjoint = false, const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<std::complex<Precision>>
            &gate_matrix = {}) {
        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        std::transform(gate_matrix.begin(), gate_matrix.end(),
                       matrix_cu.begin(), [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, params, matrix_cu);
    }

    /**
     * @brief Multi-op variant of `execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)`
     *
     * @param opNames
     * @param wires
     * @param adjoints
     * @param params
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints,
                        const std::vector<std::vector<Precision>> &params) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx],
                           params[op_idx]);
        }
    }

    /**
     * @brief Multi-op variant of `execute(const std::string &opName, const
     std::vector<int> &wires, bool adjoint = false, const std::vector<Precision>
     &params)`
     *
     * @param opNames
     * @param wires
     * @param adjoints
     * @param params
     */
    void applyOperation(const std::vector<std::string> &opNames,
                        const std::vector<std::vector<size_t>> &wires,
                        const std::vector<bool> &adjoints) {
        PL_ABORT_IF(opNames.size() != wires.size(),
                    "Incompatible number of ops and wires");
        PL_ABORT_IF(opNames.size() != adjoints.size(),
                    "Incompatible number of ops and adjoints");
        const auto num_ops = opNames.size();
        for (std::size_t op_idx = 0; op_idx < num_ops; op_idx++) {
            applyOperation(opNames[op_idx], wires[op_idx], adjoints[op_idx]);
        }
    }

    auto applyGenerator(const std::string &opName,
                        const std::vector<size_t> &wires, bool adjoint = false)
        -> PrecisionT {

        return generator_map_.at(opName)(wires, adjoint);
    }

    void applyMatrix(const std::complex<PrecisionT> *gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::string opName = {};
        size_t n = size_t{1} << wires.size();
        const std::vector<std::complex<PrecisionT>> matrix(gate_matrix,
                                                           gate_matrix + n * n);
        this->applyOperation_std(opName, wires, adjoint, {}, matrix);
    }

    void applyMatrix(const std::vector<std::complex<PrecisionT>> &gate_matrix,
                     const std::vector<size_t> &wires, bool adjoint = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        PL_ABORT_IF(gate_matrix.size() != Pennylane::Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        const std::string opName = {};
        this->applyOperation_std(opName, wires, adjoint, {}, gate_matrix);
    }

    //****************************************************************************//
    // Explicit gate calls for bindings
    //****************************************************************************//
    /* one-qubit gates */
    inline void applyIdentity(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static_cast<void>(wires);
        static_cast<void>(adjoint);
    }
    inline void applyPauliX(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliX"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliY(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyPauliZ(const std::vector<std::size_t> &wires,
                            bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyHadamard(const std::vector<std::size_t> &wires,
                              bool adjoint) {
        static const std::string name{"Hadamard"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyS(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"S"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"T"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyRX(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RX"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRY(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RY"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRZ(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RZ"}};
        applyParametricPauliGate(name, {wires.begin(), wires.end() - 1},
                                 {wires.back()}, param, adjoint);
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param0, Precision param1, Precision param2) {
        const std::string opName = "Rot";
        const std::vector<Precision> params = {param0, param1, param2};
        applyOperation(opName, wires, adjoint, params);
    }
    inline void applyRot(const std::vector<std::size_t> &wires, bool adjoint,
                         const std::vector<Precision> &params) {
        applyRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyPhaseShift(const std::vector<std::size_t> &wires,
                                bool adjoint, Precision param) {
        static const std::string name{"PhaseShift"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getPhaseShift<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }

    /* two-qubit gates */
    inline void applyCNOT(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CNOT"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CY"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"CZ"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applySWAP(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param), {},
                              wires, adjoint);
    }
    inline void applyIsingXX(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RX"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingYY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RY"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyIsingZZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          const std::vector<Precision> &params) {
        applyCRot(wires, adjoint, params[0], params[1], params[2]);
    }
    inline void applyCRot(const std::vector<std::size_t> &wires, bool adjoint,
                          Precision param0, Precision param1,
                          Precision param2) {
        const std::string opName = "CRot";
        const std::vector<Precision> params = {param0, param1, param2};
        applyOperation(opName, wires, adjoint, params);
    }

    inline void applyCRX(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRX(wires, adjoint, param);
    }
    inline void applyCRY(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRY(wires, adjoint, param);
    }
    inline void applyCRZ(const std::vector<std::size_t> &wires, bool adjoint,
                         Precision param) {
        applyRZ(wires, adjoint, param);
    }
    inline void applyControlledPhaseShift(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        applyPhaseShift(wires, adjoint, param);
    }
    inline void applySingleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        static const std::string name{"SingleExcitation"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getSingleExcitation<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void
    applySingleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationMinus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationMinus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }
    inline void applySingleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        static const std::string name{"SingleExcitationPlus"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getSingleExcitationPlus<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
    }

    /* three-qubit gates */
    inline void applyToffoli(const std::vector<std::size_t> &wires,
                             bool adjoint) {
        static const std::string name{"Toffoli"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.begin(), wires.end() - 1}, {wires.back()},
                              adjoint);
    }
    inline void applyCSWAP(const std::vector<std::size_t> &wires,
                           bool adjoint) {
        static const std::string name{"SWAP"};
        static const Precision param = 0.0;
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                              {wires.front()}, {wires.begin() + 1, wires.end()},
                              adjoint);
    }

    /* four-qubit gates */
    inline void applyDoubleExcitation(const std::vector<std::size_t> &wires,
                                      bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitation<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void
    applyDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                               bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationMinus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }
    inline void applyDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                          bool adjoint, Precision param) {
        auto &&mat = cuGates::getDoubleExcitationPlus<CFP_t>(param);
        applyDeviceMatrixGate(mat.data(), {}, wires, adjoint);
    }

    /* Multi-qubit gates */
    inline void applyMultiRZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate(names, {}, wires, param, adjoint);
    }

    /* Gate generators */
    /**
     * @brief Gradient generator function associated with the RX gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRX(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliX(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the RY gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRY(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliY(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the RZ gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRZ(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyPauliZ(wires, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorIsingXX(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingXX"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingXX<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorIsingYY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingYY"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingYY<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorIsingZZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingZZ"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingZZ<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the PhaseShift gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorPhaseShift(const std::vector<size_t> &wires,
                                               bool adj = false) {
        applyOperation("P_11", wires, adj, {0.0}, cuGates::getP11_CU<CFP_t>());
        return static_cast<PrecisionT>(1.0);
    }

    /**
     * @brief Gradient generator function associated with the controlled RX
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRX(const std::vector<size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliX(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RY
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRY(const std::vector<size_t> &wires,
                                        bool adj = false) {

        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliY(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RZ
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRZ(const std::vector<size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliZ(std::vector<size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled
     * PhaseShift gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT
    applyGeneratorControlledPhaseShift(const std::vector<size_t> &wires,
                                       bool adj = false) {
        applyOperation("P_1111", {wires}, adj, {0.0},
                       cuGates::getP1111_CU<CFP_t>());
        return static_cast<PrecisionT>(1.0);
    }

    inline PrecisionT
    applyGeneratorSingleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorSingleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorSingleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorSingleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorSingleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorSingleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorDoubleExcitation(const std::vector<std::size_t> &wires,
                                   bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitation"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitation<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorDoubleExcitationMinus(const std::vector<std::size_t> &wires,
                                        bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationMinus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationMinus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }
    inline PrecisionT
    applyGeneratorDoubleExcitationPlus(const std::vector<std::size_t> &wires,
                                       bool adjoint) {
        static const std::string name{"GeneratorDoubleExcitationPlus"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(
                gate_key, cuGates::getGeneratorDoubleExcitationPlus<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return -static_cast<PrecisionT>(0.5);
    }

    inline PrecisionT
    applyGeneratorMultiRZ(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"PauliZ"};
        static const Precision param = 0.0;
        for (const auto &w : wires) {
            applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(name, param),
                                  {}, {w}, adjoint);
        }
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Utility method for expectation value calculations.
     *
     * @param obsName String label for observable. If already exists, will used
     * cached device value. If not, `gate_matrix` is expected, and will
     * automatically cache for future reuse.
     * @param wires Target wires for expectation value.
     * @param params Parameters for a parametric gate.
     * @param gate_matrix Optional matrix for observable. Caches for future use
     * if does not exist.
     * @return auto Expectation value.
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<CFP_t> &gate_matrix = {}) -> Precision {

        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{
                      wires.rbegin(),
                      wires.rend()}; // ensure wire indexing correctly preserved
                                     // for tensor-observables

        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            gate_cache_.add_gate(obsName, par[0], gate_matrix);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        auto expect_val = getExpectationValueDeviceMatrix(
            gate_cache_.get_gate_device_ptr(obsName, par[0]), local_wires);
        return expect_val.x;
    }
    /**
     * @brief See `expval(const std::string &obsName, const std::vector<size_t>
     &wires, const std::vector<Precision> &params = {0.0}, const
     std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::string &obsName, const std::vector<size_t> &wires,
                const std::vector<Precision> &params = {0.0},
                const std::vector<std::complex<Precision>> &gate_matrix = {})
        -> Precision {
        auto &&par = (params.empty()) ? std::vector<Precision>{0.0} : params;

        std::vector<CFP_t> matrix_cu(gate_matrix.size());
        if (!(gate_cache_.gateExists(obsName, par[0]) || gate_matrix.empty())) {
            for (std::size_t i = 0; i < gate_matrix.size(); i++) {
                matrix_cu[i] = cuUtil::complexToCu<std::complex<Precision>>(
                    gate_matrix[i]);
            }
            gate_cache_.add_gate(obsName, par[0], matrix_cu);
        } else if (!gate_cache_.gateExists(obsName, par[0]) &&
                   gate_matrix.empty()) {
            std::string message =
                "Currently unsupported observable: " + obsName;
            throw LightningException(message.c_str());
        }
        return expval(obsName, wires, params, matrix_cu);
    }
    /**
     * @brief See `expval(std::vector<CFP_t> &gate_matrix = {})`
     */
    auto expval(const std::vector<size_t> &wires,
                const std::vector<std::complex<Precision>> &gate_matrix)
        -> Precision {

        PL_ABORT_IF((std::is_same<Precision,double>::value) == false, "FP32 is not supported.");

        std::vector<CFP_t> matrix_cu(gate_matrix.size());

        for (std::size_t i = 0; i < gate_matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(gate_matrix[i]);
        }

        if (gate_matrix.empty()) {
            std::string message = "Currently unsupported observable";
            throw LightningException(message.c_str());
        }

        // Wire order reversed to match expected custatevec wire ordering for
        // tensor observables.
        auto &&local_wires =
            (gate_matrix.empty())
                ? wires
                : std::vector<size_t>{wires.rbegin(), wires.rend()};

        auto expect_val =
            getExpectationValueDeviceMatrix(matrix_cu.data(), local_wires);
        return expect_val.x;
    }

    /**
     * @brief expval(H) calculation with cuSparseSpMV.
     *
     * @tparam index_type Integer type used as indices of the sparse matrix.
     * @param csr_Offsets_ptr Pointer to the array of row offsets of the sparse
     * matrix. Array of size csrOffsets_size.
     * @param csrOffsets_size Number of Row offsets of the sparse matrix.
     * @param columns_ptr Pointer to the array of column indices of the sparse
     * matrix. Array of size numNNZ
     * @param values_ptr Pointer to the array of the non-zero elements
     * @param numNNZ Number of non-zero elements.
     * @return auto Expectation value.
     */

    template <class index_type>
    auto
    getExpectationValueOnSparseSpMV(const index_type *csrOffsets_ptr,
                                    const index_type csrOffsets_size,
                                    const index_type *columns_ptr,
                                    const std::complex<Precision> *values_ptr,
                                    const index_type numNNZ) -> Precision {
        const std::size_t nIndexBits = BaseType::getNumQubits();
        const std::size_t length = std::size_t{1} << nIndexBits;

        auto device_id = BaseType::getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = BaseType::getDataBuffer().getDevTag().getStreamID();

        std::unique_ptr<DataBuffer<CFP_t>> d_sv_prime =
            std::make_unique<DataBuffer<CFP_t>>(length, device_id, stream_id,
                                                true);

        SparseMV_cuSparse<index_type, PrecisionT, CFP_t>(
            csrOffsets_ptr, csrOffsets_size, columns_ptr,
            values_ptr, numNNZ, BaseType::getData(),
            d_sv_prime->getData(), device_id, stream_id, this->getCusparseHandle());
        
        auto expect = innerProdC_CUDA(BaseType::getData(), d_sv_prime->getData(),
                                 BaseType::getLength(), device_id, stream_id,
                                 this->getCublasCaller())
                     .x;

        return expect;
    }

    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<double>
     */
    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<double>
     */
    auto probability(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        // Data return type fixed as double in custatevec function call
        std::vector<double> probabilities(Pennylane::Util::exp2(wires.size()));
        // this should be built upon by the wires not participating
        int maskLen =
            0; // static_cast<int>(BaseType::getNumQubits() - wires.size());
        int *maskBitString = nullptr; //
        int *maskOrdering = nullptr;

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<int> wires_int(wires.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            wires.begin(), wires.end(), wires_int.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumArray(
            /* custatevecHandle_t */ handle_.get(),
            /* const void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ BaseType::getNumQubits(),
            /* double* */ probabilities.data(),
            /* const int32_t* */ wires_int.data(),
            /* const uint32_t */ wires_int.size(),
            /* const int32_t* */ maskBitString,
            /* const int32_t* */ maskOrdering,
            /* const uint32_t */ maskLen));

        
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            return probabilities;
        } else {
            std::vector<PrecisionT> probs(Pennylane::Util::exp2(wires.size()));
            std::transform(
            probabilities.begin(), probabilities.end(), probs.begin(), [&](double x) {
                return static_cast<PrecisionT>(x);
            });
            return probs;
        }
    }

    /**
     * @brief Utility method for samples.
     *
     * @param num_samples Number of Samples
     *
     * @return std::vector<size_t> A 1-d array storing the samples.
     * Each sample has a length equal to the number of qubits. Each sample can
     * be accessed using the stride sample_id*num_qubits, where sample_id is a
     * number between 0 and num_samples-1.
     */
    auto generate_samples(size_t num_samples) -> std::vector<size_t> {

        std::vector<double> rand_nums(num_samples);
        custatevecSamplerDescriptor_t sampler;

        const size_t num_qubits = BaseType::getNumQubits();
        const int bitStringLen = BaseType::getNumQubits();

        std::vector<int> bitOrdering(num_qubits);
        std::iota(std::begin(bitOrdering), std::end(bitOrdering),
                  0); // Fill with 0, 1, ...,

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<Precision> dis(0.0, 1.0);
        for (size_t n = 0; n < num_samples; n++) {
            rand_nums[n] = dis(gen);
        }
        std::vector<size_t> samples(num_samples * num_qubits, 0);
        std::unordered_map<size_t, size_t> cache;
        std::vector<custatevecIndex_t> bitStrings(num_samples);

        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        // create sampler and check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerCreate(
            handle_.get(), BaseType::getData(), data_type, num_qubits, &sampler,
            num_samples, &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

        // sample preprocess
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerPreprocess(
            handle_.get(), sampler, extraWorkspace, extraWorkspaceSizeInBytes));

        // sample bit strings
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerSample(
            handle_.get(), sampler, bitStrings.data(), bitOrdering.data(),
            bitStringLen, rand_nums.data(), num_samples,
            CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

        // destroy descriptor and handle
        PL_CUSTATEVEC_IS_SUCCESS(custatevecSamplerDestroy(sampler));

        // Pick samples
        for (size_t i = 0; i < num_samples; i++) {
            auto idx = bitStrings[i];
            // If cached, retrieve sample from cache
            if (cache.count(idx) != 0) {
                size_t cache_id = cache[idx];
                auto it_temp = samples.begin() + cache_id * num_qubits;
                std::copy(it_temp, it_temp + num_qubits,
                          samples.begin() + i * num_qubits);
            }
            // If not cached, compute
            else {
                for (size_t j = 0; j < num_qubits; j++) {
                    samples[i * num_qubits + (num_qubits - 1 - j)] =
                        (idx >> j) & 1U;
                }
                cache[idx] = i;
            }
        }

        if (extraWorkspaceSizeInBytes > 0)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));

        return samples;
    }

    /**
     * @brief Get expectation value for a sum of Pauli words.
     *
     * @param pauli_words Vector of Pauli-words to evaluate expectation value.
     * @param tgts Coupled qubit index to apply each Pauli term.
     * @param coeffs Numpy array buffer of size |pauli_words|
     * @return auto Expectation value.
     */
    auto getExpectationValuePauliWords(
        const std::vector<std::string> &pauli_words,
        const std::vector<std::vector<std::size_t>> &tgts,
        const std::complex<Precision> *coeffs) {

        uint32_t nIndexBits = static_cast<uint32_t>(BaseType::getNumQubits());
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        // Note: due to API design, cuStateVec assumes this is always a double.
        // Push NVIDIA to move this to behind API for future releases, and
        // support 32/64 bits.
        std::vector<double> expect(pauli_words.size());

        std::vector<std::vector<custatevecPauli_t>> pauliOps;

        std::vector<custatevecPauli_t *> pauliOps_ptr;

        for (auto &p_word : pauli_words) {
            pauliOps.push_back(cuUtil::pauliStringToEnum(p_word));
            pauliOps_ptr.push_back((*pauliOps.rbegin()).data());
        }

        std::vector<std::vector<int32_t>> basisBits;
        std::vector<int32_t *> basisBits_ptr;
        std::vector<uint32_t> n_basisBits;

        for (auto &wires : tgts) {
            std::vector<int32_t> wiresInt(wires.size());
            std::transform(wires.begin(), wires.end(), wiresInt.begin(),
                           [&](std::size_t x) {
                               return static_cast<int>(
                                   BaseType::getNumQubits() - 1 - x);
                           });
            basisBits.push_back(wiresInt);
            basisBits_ptr.push_back((*basisBits.rbegin()).data());
            n_basisBits.push_back(wiresInt.size());
        }

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationsOnPauliBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double* */ expect.data(),
            /* const custatevecPauli_t ** */
            const_cast<const custatevecPauli_t **>(pauliOps_ptr.data()),
            /* const uint32_t */ static_cast<uint32_t>(pauliOps.size()),
            /* const int32_t ** */
            const_cast<const int32_t **>(basisBits_ptr.data()),
            /* const uint32_t */ n_basisBits.data()));

        std::complex<Precision> result{0, 0};

        if constexpr (std::is_same_v<Precision, double>) {
            for (std::size_t idx = 0; idx < expect.size(); idx++) {
                result += expect[idx] * coeffs[idx];
            }
            return std::real(result);
        } else {
            std::vector<Precision> expect_cast(expect.size());
            std::transform(expect.begin(), expect.end(), expect_cast.begin(),
                           [](double x) { return static_cast<float>(x); });

            for (std::size_t idx = 0; idx < expect_cast.size(); idx++) {
                result += expect_cast[idx] * coeffs[idx];
            }

            return std::real(result);
        }
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
     * @brief Get the cuSPARSE handle that the object is using.
     *
     * @return cusparseHandle_t returns the cuSPARSE handle.
     */
    auto getCusparseHandle() const -> cusparseHandle_t {
        if (!cusparsehandle_)
            cusparsehandle_ = make_shared_cusparse_handle();
        return cusparsehandle_.get();
    }

    /**
     * @brief Get the cuStateVec handle that the object is using.
     *
     * @return custatevecHandle_t returns the cuStateVec handle.
     */
    auto getCusvHandle() const -> custatevecHandle_t { return handle_.get(); }

    /**
     * @brief Get a host data copy.
     *
     * @return std::vector<std::complex<PrecisionT>>
     */

    auto getDataVector() -> std::vector<std::complex<PrecisionT>> {
        std::vector<std::complex<PrecisionT>> data_host(BaseType::getLength());
        this->CopyGpuDataToHost(data_host.data(), data_host.size());
        return data_host;
    }

  private:
    SharedCusvHandle handle_;
    SharedCublasCaller cublascaller_;
    mutable SharedCusparseHandle
        cusparsehandle_; // This member is mutable to allow lazy initialization.
    GateCache<Precision> gate_cache_;
    using ParFunc = std::function<void(const std::vector<size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using GeneratorFunc =
        std::function<Precision(const std::vector<size_t> &, bool)>;

    using FMap = std::unordered_map<std::string, ParFunc>;
    using GMap = std::unordered_map<std::string, GeneratorFunc>;

    const FMap par_gates_{
        {"RX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRX(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRY(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRZ(std::forward<decltype(wires)>(wires),
                     std::forward<decltype(adjoint)>(adjoint),
                     std::forward<decltype(params[0])>(params[0]));
         }},
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyPhaseShift(std::forward<decltype(wires)>(wires),
                             std::forward<decltype(adjoint)>(adjoint),
                             std::forward<decltype(params[0])>(params[0]));
         }},
        {"MultiRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyMultiRZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingXX(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingYY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingYY(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingZZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingZZ(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRX",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRX(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRY(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRZ(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applySingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint),
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"Rot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRot(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params)>(params));
         }},
        {"CRot", [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRot(std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
         }}};

    const std::unordered_map<std::string, custatevecPauli_t> native_gates_{
        {"RX", CUSTATEVEC_PAULI_X},       {"RY", CUSTATEVEC_PAULI_Y},
        {"RZ", CUSTATEVEC_PAULI_Z},       {"CRX", CUSTATEVEC_PAULI_X},
        {"CRY", CUSTATEVEC_PAULI_Y},      {"CRZ", CUSTATEVEC_PAULI_Z},
        {"Identity", CUSTATEVEC_PAULI_I}, {"I", CUSTATEVEC_PAULI_I}};

    // Holds the mapping from gate labels to associated generator functions.
    const GMap generator_map_{

        {"RX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRX(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"RY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRY(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"RZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorRZ(std::forward<decltype(wires)>(wires),
                                     std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingXX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingXX(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingYY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingYY(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"IsingZZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingZZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRX",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRX(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRY(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"CRZ",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorCRZ(std::forward<decltype(wires)>(wires),
                                      std::forward<decltype(adjoint)>(adjoint));
         }},
        {"PhaseShift",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"ControlledPhaseShift",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorControlledPhaseShift(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitation",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitationMinus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"SingleExcitationPlus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorSingleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitation",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitation(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitationMinus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
        {"DoubleExcitationPlus",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorDoubleExcitationPlus(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},

        {"MultiRZ", [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorMultiRZ(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }}};

    /**
     * @brief Normalize the index ordering to match PennyLane.
     *
     * @tparam IndexType Integer value type.
     * @param indices Given indices to transform.
     */
    template <typename IndexType>
    inline auto NormalizeIndices(std::vector<IndexType> indices)
        -> std::vector<IndexType> {
        std::vector<IndexType> t_indices(std::move(indices));
        std::transform(t_indices.begin(), t_indices.end(), t_indices.begin(),
                       [&](IndexType i) -> IndexType {
                           return BaseType::getNumQubits() - 1 - i;
                       });
        return t_indices;
    }

    /**
     * @brief Apply parametric Pauli gates using custateVec calls.
     *
     * @param angle Rotation angle.
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyParametricPauliGate(const std::vector<std::string> &pauli_words,
                                  std::vector<std::size_t> ctrls,
                                  std::vector<std::size_t> tgts,
                                  Precision param, bool use_adjoint = false) {
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        // Transform indices between PL & cuQuantum ordering
        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<custatevecPauli_t> pauli_enums;
        pauli_enums.reserve(pauli_words.size());
        for (const auto &pauli_str : pauli_words) {
            pauli_enums.push_back(native_gates_.at(pauli_str));
        }
        const auto local_angle = (use_adjoint) ? param / 2 : -param / 2;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyPauliRotation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* double */ local_angle,
            /* const custatevecPauli_t* */ pauli_enums.data(),
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size()));
    }

    /**
     * @brief Apply a given host or device-stored array representing the gate
     * `matrix` to the state vector at qubit indices given by `tgts` and
     * control-lines given by `ctrls`. The adjoint can be taken by setting
     * `use_adjoint` to true.
     *
     * @param matrix Host- or device data array in row-major order representing
     * a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyDeviceMatrixGate(const CFP_t *matrix,
                               const std::vector<std::size_t> &ctrls,
                               const std::vector<std::size_t> &tgts,
                               bool use_adjoint = false) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrixGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const uint32_t */ tgts.size(),
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // apply gate
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrix(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }

    /**
     * @brief Apply a given host-matrix `matrix` to the state vector at qubit
     * indices given by `tgts` and control-lines given by `ctrls`. The adjoint
     * can be taken by setting `use_adjoint` to true.
     *
     * @param matrix Host-data vector in row-major order of a given gate.
     * @param ctrls Control line qubits.
     * @param tgts Target qubits.
     * @param use_adjoint Use adjoint of given gate.
     */
    void applyHostMatrixGate(const std::vector<CFP_t> &matrix,
                             const std::vector<std::size_t> &ctrls,
                             const std::vector<std::size_t> &tgts,
                             bool use_adjoint = false) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;
        int nIndexBits = BaseType::getNumQubits();

        std::vector<int> ctrlsInt(ctrls.size());
        std::vector<int> tgtsInt(tgts.size());

        std::transform(
            ctrls.begin(), ctrls.end(), ctrlsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrixGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const uint32_t */ tgts.size(),
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        // allocate external workspace if necessary
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        // apply gate
        PL_CUSTATEVEC_IS_SUCCESS(custatevecApplyMatrix(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t */ use_adjoint,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ nullptr,
            /* const uint32_t */ ctrls.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
    }
    void applyHostMatrixGate(const std::vector<std::complex<Precision>> &matrix,
                             const std::vector<std::size_t> &ctrls,
                             const std::vector<std::size_t> &tgts,
                             bool use_adjoint = false) {
        std::vector<CFP_t> matrix_cu(matrix.size());
        for (std::size_t i = 0; i < matrix.size(); i++) {
            matrix_cu[i] =
                cuUtil::complexToCu<std::complex<Precision>>(matrix[i]);
        }

        applyHostMatrixGate(matrix_cu, ctrls, tgts, use_adjoint);
    }

    /**
     * @brief Get expectation of a given host-defined matrix.
     *
     * @param matrix Host-defined row-major order gate matrix.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueHostMatrix(const std::vector<CFP_t> &matrix,
                                       const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix.data(),
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgts.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        return expect;
    }

    /**
     * @brief Get expectation of a given host or device defined array.
     *
     * @param matrix Host or device defined row-major order gate matrix array.
     * @param tgts Target qubits.
     * @return auto Expectation value.
     */
    auto getExpectationValueDeviceMatrix(const CFP_t *matrix,
                                         const std::vector<std::size_t> &tgts) {
        void *extraWorkspace = nullptr;
        size_t extraWorkspaceSizeInBytes = 0;

        std::vector<int> tgtsInt(tgts.size());
        std::transform(
            tgts.begin(), tgts.end(), tgtsInt.begin(), [&](std::size_t x) {
                return static_cast<int>(BaseType::getNumQubits() - 1 - x);
            });

        size_t nIndexBits = BaseType::getNumQubits();
        cudaDataType_t data_type;
        custatevecComputeType_t compute_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
            compute_type = CUSTATEVEC_COMPUTE_64F;
        } else {
            data_type = CUDA_C_32F;
            compute_type = CUSTATEVEC_COMPUTE_32F;
        }

        // check the size of external workspace
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectationGetWorkspaceSize(
            /* custatevecHandle_t */ handle_.get(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* size_t* */ &extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }

        CFP_t expect;

        // compute expectation
        PL_CUSTATEVEC_IS_SUCCESS(custatevecComputeExpectation(
            /* custatevecHandle_t */ handle_.get(),
            /* void* */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t */ nIndexBits,
            /* void* */ &expect,
            /* cudaDataType_t */ data_type,
            /* double* */ nullptr,
            /* const void* */ matrix,
            /* cudaDataType_t */ data_type,
            /* custatevecMatrixLayout_t */ CUSTATEVEC_MATRIX_LAYOUT_ROW,
            /* const int32_t* */ tgtsInt.data(),
            /* const uint32_t */ tgtsInt.size(),
            /* custatevecComputeType_t */ compute_type,
            /* void* */ extraWorkspace,
            /* size_t */ extraWorkspaceSizeInBytes));

        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        return expect;
    }
};
}; // namespace Pennylane::LightningGPU
