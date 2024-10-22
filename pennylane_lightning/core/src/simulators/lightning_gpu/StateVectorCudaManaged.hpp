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

#include "CPUMemoryModel.hpp"

#include "cuError.hpp"
#include "cuStateVecError.hpp"
#include "cuStateVec_helpers.hpp"

#include "LinearAlg.hpp"
#include "Util.hpp"

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
                                std::size_t thread_per_block,
                                cudaStream_t stream_id);
extern void setStateVector_CUDA(cuDoubleComplex *sv, long &num_indices,
                                cuDoubleComplex *value, long *indices,
                                std::size_t thread_per_block,
                                cudaStream_t stream_id);

extern void setBasisState_CUDA(cuComplex *sv, cuComplex &value,
                               const std::size_t index, bool async,
                               cudaStream_t stream_id);
extern void setBasisState_CUDA(cuDoubleComplex *sv, cuDoubleComplex &value,
                               const std::size_t index, bool async,
                               cudaStream_t stream_id);

/**
 * @brief Managed memory CUDA state-vector class using custateVec backed
 * gate-calls.
 *
 * @tparam Precision Floating-point precision type.
 */
template <class Precision = double>
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
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Undefined;

    StateVectorCudaManaged() = delete;
    StateVectorCudaManaged(std::size_t num_qubits)
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits),
          handle_(make_shared_cusv_handle()),
          cublascaller_(make_shared_cublas_caller()), gate_cache_(true) {
        resetStateVector();
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
    };

    StateVectorCudaManaged(
        std::size_t num_qubits, const DevTag<int> &dev_tag, bool alloc = true,
        SharedCusvHandle cusvhandle_in = make_shared_cusv_handle(),
        SharedCublasCaller cublascaller_in = make_shared_cublas_caller(),
        SharedCusparseHandle cusparsehandle_in = make_shared_cusparse_handle())
        : StateVectorCudaBase<Precision, StateVectorCudaManaged<Precision>>(
              num_qubits, dev_tag, alloc),
          handle_(std::move(cusvhandle_in)),
          cublascaller_(std::move(cublascaller_in)),
          cusparsehandle_(std::move(cusparsehandle_in)),
          gate_cache_(true, dev_tag) {
        resetStateVector();
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
    };

    StateVectorCudaManaged(const CFP_t *gpu_data, std::size_t length)
        : StateVectorCudaManaged(Pennylane::Util::log2(length)) {
        BaseType::CopyGpuDataToGpuIn(gpu_data, length, false);
    }

    StateVectorCudaManaged(
        const CFP_t *gpu_data, std::size_t length, DevTag<int> dev_tag,
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
                           std::size_t length)
        : StateVectorCudaManaged(Pennylane::Util::log2(length)) {
        BaseType::CopyHostDataToGpu(host_data, length, false);
    }

    StateVectorCudaManaged(std::complex<Precision> *host_data,
                           std::size_t length)
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
     * @brief the statevector data to the |0...0> state.
     * @param use_async Use an asynchronous memory copy or not. Default is
     * false.
     */
    void resetStateVector(bool use_async = false) {
        BaseType::getDataBuffer().zeroInit();
        std::size_t index = 0;
        ComplexT value(1.0, 0.0);
        setBasisState_(value, index, use_async);
    };

    /**
     * @brief Prepare a single computational basis state.
     *
     * @param state Binary number representing the index
     * @param wires Wires.
     * @param use_async(Optional[bool]): immediately sync with host-sv after
     * applying operation.
     */
    void setBasisState(const std::vector<std::size_t> &state,
                       const std::vector<std::size_t> &wires,
                       const bool use_async = false) {
        PL_ABORT_IF_NOT(state.size() == wires.size(),
                        "state and wires must have equal dimensions.");
        const auto num_qubits = BaseType::getNumQubits();
        PL_ABORT_IF_NOT(
            std::find_if(wires.begin(), wires.end(),
                         [&num_qubits](const auto i) {
                             return i >= num_qubits;
                         }) == wires.end(),
            "wires must take values lower than the number of qubits.");
        const auto n_wires = wires.size();
        std::size_t index{0U};
        for (std::size_t k = 0; k < n_wires; k++) {
            index |= state[k] << (num_qubits - 1 - wires[k]);
        }

        const std::complex<PrecisionT> value(1.0, 0.0);

        BaseType::getDataBuffer().zeroInit();
        setBasisState_(value, index, use_async);
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param state_ptr Pointer to the initial state data.
     * @param num_states Length of the initial state data.
     * @param wires Wires.
     * @param use_async Use an asynchronous memory copy. Default is false.
     */
    void setStateVector(const ComplexT *state_ptr, const std::size_t num_states,
                        const std::vector<std::size_t> &wires,
                        bool use_async = false) {
        PL_ABORT_IF_NOT(num_states == Pennylane::Util::exp2(wires.size()),
                        "Inconsistent state and wires dimensions.");

        const auto num_qubits = BaseType::getNumQubits();

        PL_ABORT_IF_NOT(std::find_if(wires.begin(), wires.end(),
                                     [&num_qubits](const auto i) {
                                         return i >= num_qubits;
                                     }) == wires.end(),
                        "Invalid wire index.");

        using index_type =
            typename std::conditional<std::is_same<PrecisionT, float>::value,
                                      int32_t, int64_t>::type;

        // Calculate the indices of the state-vector to be set.
        // TODO: Could move to GPU calculation if the state size is large.
        std::vector<index_type> indices(num_states);
        const std::size_t num_wires = wires.size();
        constexpr std::size_t one{1U};
        for (std::size_t i = 0; i < num_states; i++) {
            std::size_t index{0U};
            for (std::size_t j = 0; j < num_wires; j++) {
                const std::size_t bit = (i & (one << j)) >> j;
                index |= bit << (num_qubits - 1 - wires[num_wires - 1 - j]);
            }
            indices[i] = static_cast<index_type>(index);
        }
        setStateVector_<index_type>(num_states, state_ptr, indices.data(),
                                    use_async);
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
     * @param matrix Gate data (in row-major format).
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires, bool adjoint,
                        const std::vector<Precision> &params,
                        const std::vector<ComplexT> &matrix) {
        std::vector<CFP_t> matrix_cu(matrix.size());
        std::transform(matrix.begin(), matrix.end(), matrix_cu.begin(),
                       [](const std::complex<Precision> &x) {
                           return cuUtil::complexToCu<std::complex<Precision>>(
                               x);
                       });
        applyOperation(opName, wires, adjoint, params, matrix_cu);
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
     * @param gate_matrix Gate data (in row-major format).
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool adjoint = false,
                        const std::vector<Precision> &params = {0.0},
                        const std::vector<CFP_t> &gate_matrix = {}) {
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
        } else if (opName == "GlobalPhase") {
            PrecisionT param = adjoint ? -params[0] : params[0];
            CFP_t scale_factor{std::cos(param), -std::sin(param)};
            scaleC_CUDA<CFP_t, CFP_t, int>(
                scale_factor, BaseType::getDataBuffer().getData(),
                BaseType::getDataBuffer().getLength(),
                BaseType::getDataBuffer().getDevTag().getDeviceID(),
                BaseType::getDataBuffer().getDevTag().getStreamID(),
                getCublasCaller());
        } else if (native_gates_.find(opName) != native_gates_.end()) {
            applyParametricPauliGate_({opName}, ctrls, tgts, params.front(),
                                      adjoint);
        } else if (opName == "Rot" || opName == "CRot") {
            if (adjoint) {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[2], params[1], params[0]);
                applyDeviceMatrixGate(rot_matrix.data(), ctrls, tgts, true);
            } else {
                auto rot_matrix =
                    cuGates::getRot<CFP_t>(params[0], params[1], params[2]);
                applyDeviceMatrixGate(rot_matrix.data(), ctrls, tgts, false);
            }
        } else if (opName == "Matrix") {
            DataBuffer<CFP_t, int> d_matrix{
                gate_matrix.size(), BaseType::getDataBuffer().getDevTag(),
                true};
            d_matrix.CopyHostDataToGpu(gate_matrix.data(), d_matrix.getLength(),
                                       false);
            // ensure wire indexing correctly preserved for tensor-observables
            const std::vector<std::size_t> ctrls_local{ctrls.rbegin(),
                                                       ctrls.rend()};
            const std::vector<std::size_t> tgts_local{tgts.rbegin(),
                                                      tgts.rend()};
            applyDeviceMatrixGate(d_matrix.getData(), ctrls_local, tgts_local,
                                  adjoint);
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
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param tgt_wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(
        const std::string &opName,
        const std::vector<std::size_t> &controlled_wires,
        const std::vector<bool> &controlled_values,
        const std::vector<std::size_t> &tgt_wires, bool adjoint = false,
        const std::vector<Precision> &params = {0.0},
        [[maybe_unused]] const std::vector<ComplexT> &gate_matrix = {}) {
        PL_ABORT_IF_NOT(opName == "GlobalPhase",
                        "Only GlobalPhase gate is supported.");
        PL_ABORT_IF(controlled_wires.size() != controlled_values.size(),
                    "`ctrls` and `ctrls_values` must have the same size.");
        auto ctrlsInt = NormalizeCastIndices<std::size_t, int>(
            controlled_wires, BaseType::getNumQubits());
        auto tgtsInt = NormalizeCastIndices<std::size_t, int>(
            tgt_wires, BaseType::getNumQubits());
        auto ctrls_valuesInt =
            Pennylane::Util::cast_vector<bool, int>(controlled_values);

        if (opName == "GlobalPhase") {
            const std::vector<std::string> names(tgt_wires.size(), "I");
            applyParametricPauliGeneralGate_(names, ctrlsInt, ctrls_valuesInt,
                                             tgtsInt, 2 * params[0], adjoint);
        }
    }

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool adjoint = false) -> PrecisionT {
        auto it = generator_map_.find(opName);
        PL_ABORT_IF(it == generator_map_.end(), "Unsupported generator!");
        return (it->second)(wires, adjoint);
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
                     const std::vector<std::size_t> &wires,
                     bool adjoint = false) {
        PL_ABORT_IF(wires.empty(), "Number of wires must be larger than 0");
        const std::string opName = "Matrix";
        std::size_t n = std::size_t{1} << wires.size();
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
                     const std::vector<std::size_t> &wires,
                     bool adjoint = false) {
        PL_ABORT_IF(gate_matrix.size() !=
                        Pennylane::Util::exp2(2 * wires.size()),
                    "The size of matrix does not match with the given "
                    "number of wires");
        applyMatrix(gate_matrix.data(), wires, adjoint);
    }

    /**
     * @brief Collapse the state vector after having measured one of the qubit.
     *
     * Note: The branch parameter imposes the measurement result on the given
     * wire.
     *
     * @param wire Wire to measure.
     * @param branch Branch 0 or 1.
     */
    void collapse(std::size_t wire, bool branch) {
        PL_ABORT_IF_NOT(wire < BaseType::getNumQubits(), "Invalid wire index.");
        cudaDataType_t data_type;

        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type = CUDA_C_64F;
        } else {
            data_type = CUDA_C_32F;
        }

        std::vector<int> basisBits(1, BaseType::getNumQubits() - 1 - wire);

        double abs2sum0;
        double abs2sum1;

        PL_CUSTATEVEC_IS_SUCCESS(custatevecAbs2SumOnZBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void *sv */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t nIndexBits */ BaseType::getNumQubits(),
            /* double * */ &abs2sum0,
            /* double * */ &abs2sum1,
            /* const int32_t * */ basisBits.data(),
            /* const uint32_t nBasisBits */ basisBits.size()));

        const double norm = branch ? abs2sum1 : abs2sum0;

        const int parity = static_cast<int>(branch);

        PL_CUSTATEVEC_IS_SUCCESS(custatevecCollapseOnZBasis(
            /* custatevecHandle_t */ handle_.get(),
            /* void *sv */ BaseType::getData(),
            /* cudaDataType_t */ data_type,
            /* const uint32_t nIndexBits */ BaseType::getNumQubits(),
            /* const int32_t parity */ parity,
            /* const int32_t *basisBits */ basisBits.data(),
            /* const uint32_t nBasisBits */ basisBits.size(),
            /* double norm */ norm));
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
        applyParametricPauliGate_(name, {wires.begin(), wires.end() - 1},
                                  {wires.back()}, param, adjoint);
    }
    inline void applyRY(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RY"}};
        applyParametricPauliGate_(name, {wires.begin(), wires.end() - 1},
                                  {wires.back()}, param, adjoint);
    }
    inline void applyRZ(const std::vector<std::size_t> &wires, bool adjoint,
                        Precision param) {
        static const std::vector<std::string> name{{"RZ"}};
        applyParametricPauliGate_(name, {wires.begin(), wires.end() - 1},
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
        applyParametricPauliGate_(names, {}, wires, param, adjoint);
    }
    inline void applyIsingYY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RY"});
        applyParametricPauliGate_(names, {}, wires, param, adjoint);
    }
    inline void applyIsingZZ(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::vector<std::string> names(wires.size(), {"RZ"});
        applyParametricPauliGate_(names, {}, wires, param, adjoint);
    }
    inline void applyIsingXY(const std::vector<std::size_t> &wires,
                             bool adjoint, Precision param) {
        static const std::string name{"IsingXY"};
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key, cuGates::getIsingXY<CFP_t>(param));
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
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
        applyParametricPauliGate_(names, {}, wires, param, adjoint);
    }

    /* Gate generators */
    /**
     * @brief Gradient generator function associated with the GlobalPhase gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorGlobalPhase(
        [[maybe_unused]] const std::vector<std::size_t> &wires,
        [[maybe_unused]] bool adj = false) {
        return static_cast<PrecisionT>(-1.0);
    }

    /* Gate generators */
    /**
     * @brief Gradient generator function associated with the RX gate.
     *
     * @param sv Statevector
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorRX(const std::vector<std::size_t> &wires,
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
    inline PrecisionT applyGeneratorRY(const std::vector<std::size_t> &wires,
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
    inline PrecisionT applyGeneratorRZ(const std::vector<std::size_t> &wires,
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

    inline PrecisionT
    applyGeneratorIsingXY(const std::vector<std::size_t> &wires, bool adjoint) {
        static const std::string name{"GeneratorIsingXY"};
        static const Precision param = 0.0;
        const auto gate_key = std::make_pair(name, param);
        if (!gate_cache_.gateExists(gate_key)) {
            gate_cache_.add_gate(gate_key,
                                 cuGates::getGeneratorIsingXY<CFP_t>());
        }
        applyDeviceMatrixGate(gate_cache_.get_gate_device_ptr(gate_key), {},
                              wires, adjoint);
        return static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the PhaseShift gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT
    applyGeneratorPhaseShift(const std::vector<std::size_t> &wires,
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
    inline PrecisionT applyGeneratorCRX(const std::vector<std::size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliX(std::vector<std::size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RY
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRY(const std::vector<std::size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliY(std::vector<std::size_t>{wires.back()}, adj);
        return -static_cast<PrecisionT>(0.5);
    }

    /**
     * @brief Gradient generator function associated with the controlled RZ
     * gate.
     *
     * @param wires Wires to apply operation.
     * @param adj Takes adjoint of operation if true. Defaults to false.
     */
    inline PrecisionT applyGeneratorCRZ(const std::vector<std::size_t> &wires,
                                        bool adj = false) {
        applyOperation("P_11", {wires.front()}, adj, {0.0},
                       cuGates::getP11_CU<CFP_t>());
        applyPauliZ(std::vector<std::size_t>{wires.back()}, adj);
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
    applyGeneratorControlledPhaseShift(const std::vector<std::size_t> &wires,
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
        BaseType::CopyGpuDataToHost(data_host.data(), data_host.size());
        return data_host;
    }

  private:
    SharedCusvHandle handle_;
    SharedCublasCaller cublascaller_;
    mutable SharedCusparseHandle
        cusparsehandle_; // This member is mutable to allow lazy initialization.
    GateCache<Precision> gate_cache_;
    using ParFunc = std::function<void(const std::vector<std::size_t> &, bool,
                                       const std::vector<Precision> &)>;
    using GeneratorFunc =
        std::function<Precision(const std::vector<std::size_t> &, bool)>;

    using FMap = std::unordered_map<std::string, ParFunc>;
    using GMap = std::unordered_map<std::string, GeneratorFunc>;

    const FMap par_gates_{
        // LCOV_EXCL_START
        // Calculation passed to applyParametricPauliGate
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
        // LCOV_EXCL_STOP
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
        {"IsingXY",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyIsingXY(std::forward<decltype(wires)>(wires),
                          std::forward<decltype(adjoint)>(adjoint),
                          std::forward<decltype(params[0])>(params[0]));
         }},
        // LCOV_EXCL_START
        // Calculation passed to applyParametricPauliGate
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
        // LCOV_EXCL_STOP
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
        // LCOV_EXCL_START
        {"Rot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyRot(std::forward<decltype(wires)>(wires),
                      std::forward<decltype(adjoint)>(adjoint),
                      std::forward<decltype(params)>(params));
         }},
        {"CRot",
         [&](auto &&wires, auto &&adjoint, auto &&params) {
             applyCRot(std::forward<decltype(wires)>(wires),
                       std::forward<decltype(adjoint)>(adjoint),
                       std::forward<decltype(params)>(params));
         }}
        // LCOV_EXCL_STOP
    };

    const std::unordered_map<std::string, custatevecPauli_t> native_gates_{
        {"RX", CUSTATEVEC_PAULI_X},       {"RY", CUSTATEVEC_PAULI_Y},
        {"RZ", CUSTATEVEC_PAULI_Z},       {"CRX", CUSTATEVEC_PAULI_X},
        {"CRY", CUSTATEVEC_PAULI_Y},      {"CRZ", CUSTATEVEC_PAULI_Z},
        {"Identity", CUSTATEVEC_PAULI_I}, {"I", CUSTATEVEC_PAULI_I}};

    // Holds the mapping from gate labels to associated generator functions.
    const GMap generator_map_{
        {"GlobalPhase",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorGlobalPhase(
                 std::forward<decltype(wires)>(wires),
                 std::forward<decltype(adjoint)>(adjoint));
         }},
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
        {"IsingXY",
         [&](auto &&wires, auto &&adjoint) {
             return applyGeneratorIsingXY(
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

    /** @brief Set value for a single element of the state-vector on device.
     * This method is implemented by cudaMemcpy.
     *
     * @param value Value to be set for the target element.
     * @param index Index of the target element.
     * @param async Use an asynchronous memory copy.
     */
    void setBasisState_(const std::complex<Precision> &value,
                        const std::size_t index, const bool async = false) {
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
    template <class index_type, std::size_t thread_per_block = 256>
    void setStateVector_(const index_type num_indices,
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
        PL_CUDA_IS_SUCCESS(cudaDeviceSynchronize());
    }

    /**
     * @brief Apply parametric Pauli gates using custateVec calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrls Control wires
     * @param tgts target wires.
     * @param params Rotation parameters.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyParametricPauliGate_(const std::vector<std::string> &pauli_words,
                                   std::vector<std::size_t> ctrls,
                                   std::vector<std::size_t> tgts,
                                   Precision param, bool use_adjoint = false) {
        // Transform indices between PL & cuQuantum ordering
        auto ctrlsInt = NormalizeCastIndices<std::size_t, int>(
            ctrls, BaseType::getNumQubits());
        auto tgtsInt = NormalizeCastIndices<std::size_t, int>(
            tgts, BaseType::getNumQubits());

        const std::vector<int> ctrls_valuesInt(ctrls.size(), 1);

        applyParametricPauliGeneralGate_(pauli_words, ctrlsInt, ctrls_valuesInt,
                                         tgtsInt, param, use_adjoint);
    }

    /**
     * @brief Apply a parametric Pauli gate using custateVec calls.
     *
     * @param pauli_words List of Pauli words representing operation.
     * @param ctrlsInt Control wires
     * @param ctrls_valuesInt Control values
     * @param tgtsInt target wires.
     * @param param Rotation angle.
     * @param use_adjoint Take adjoint of operation.
     */
    void applyParametricPauliGeneralGate_(
        const std::vector<std::string> &pauli_words,
        const std::vector<int> &ctrlsInt,
        const std::vector<int> &ctrls_valuesInt, const std::vector<int> tgtsInt,
        Precision param, bool use_adjoint = false) {
        int nIndexBits = BaseType::getNumQubits();

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
            /* const uint32_t */ tgtsInt.size(),
            /* const int32_t* */ ctrlsInt.data(),
            /* const int32_t* */ ctrls_valuesInt.data(),
            /* const uint32_t */ ctrlsInt.size()));
        PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(
            BaseType::getDataBuffer().getDevTag().getStreamID()));
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
        std::size_t extraWorkspaceSizeInBytes = 0;
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
            /* std::size_t* */ &extraWorkspaceSizeInBytes));

        PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(
            BaseType::getDataBuffer().getDevTag().getStreamID()));

        // allocate external workspace if necessary
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes > 0) {
            PL_CUDA_IS_SUCCESS(
                cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
        }
        // LCOV_EXCL_STOP

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
            /* std::size_t */ extraWorkspaceSizeInBytes));

        PL_CUDA_IS_SUCCESS(cudaStreamSynchronize(
            BaseType::getDataBuffer().getDevTag().getStreamID()));
        // LCOV_EXCL_START
        if (extraWorkspaceSizeInBytes)
            PL_CUDA_IS_SUCCESS(cudaFree(extraWorkspace));
        // LCOV_EXCL_STOP
    }
};
}; // namespace Pennylane::LightningGPU
