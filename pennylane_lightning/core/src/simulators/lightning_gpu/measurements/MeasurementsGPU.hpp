// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
 * @file
 * Defines a class for the measurement of observables in quantum states
 * represented by a Lightning Qubit StateVector class.
 */

#pragma once

#include <algorithm>
#include <complex>
#include <cuda.h>
#include <cusparse.h>
#include <custatevec.h> // custatevecApplyMatrix
#include <random>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "MeasurementsBase.hpp"
#include "Observables.hpp"
#include "ObservablesGPU.hpp"
#include "StateVectorCudaManaged.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Measures;
using namespace Pennylane::Observables;
using namespace Pennylane::LightningGPU::Observables;
namespace cuUtil = Pennylane::LightningGPU::Util;
using Pennylane::LightningGPU::StateVectorCudaManaged;
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::Measures {
/**
 * @brief Observable's Measurement Class.
 *
 * This class couples with a statevector to performs measurements.
 * Observables are defined by its operator(matrix), the observable class,
 * or through a string-based function dispatch.
 *
 * @tparam StateVectorT type of the statevector to be measured.
 */
template <class StateVectorT>
class Measurements final
    : public MeasurementsBase<StateVectorT, Measurements<StateVectorT>> {
  private:
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using BaseType = MeasurementsBase<StateVectorT, Measurements<StateVectorT>>;
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    cudaDataType_t data_type_;

  public:
    explicit Measurements(StateVectorT &statevector) : BaseType{statevector} {
        if constexpr (std::is_same_v<CFP_t, cuDoubleComplex> ||
                      std::is_same_v<CFP_t, double2>) {
            data_type_ = CUDA_C_64F;
        } else {
            data_type_ = CUDA_C_32F;
        }
    };

    /**
     * @brief Utility method for probability calculation using given wires.
     *
     * @param wires List of wires to return probabilities for in lexicographical
     * order.
     * @return std::vector<PrecisionT>
     */
    auto probs(const std::vector<size_t> &wires) -> std::vector<PrecisionT> {
        return this->_statevector.probability(wires);
    }

    /**
     * @brief Utility method for probability calculation for a full wires.
     *
     * @return std::vector<PrecisionT>
     */
    auto probs() -> std::vector<PrecisionT> {
        std::vector<size_t> wires;
        for (size_t i = 0; i < this->_statevector.getNumQubits(); i++) {
            wires.push_back(i);
        }
        return this->probs(wires);
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
        return this->_statevector.generate_samples(num_samples);
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
    auto expval(const index_type *csrOffsets_ptr,
                const index_type csrOffsets_size, const index_type *columns_ptr,
                const std::complex<PrecisionT> *values_ptr,
                const index_type numNNZ) -> PrecisionT {
        return this->_statevector
            .template getExpectationValueOnSparseSpMV<index_type>(
                csrOffsets_ptr, csrOffsets_size, columns_ptr, values_ptr,
                numNNZ);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::string &operation, const std::vector<size_t> &wires)
        -> PrecisionT {
        std::vector<PrecisionT> params = {0.0};
        std::vector<ComplexT> gate_matrix = {};
        return this->_statevector.expval(operation, wires, params, gate_matrix);
    }

    /**
     * @brief Expected value for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with expected values for the
     * observables.
     */
    auto expval(const std::vector<std::string> &operations_list,
                const std::vector<std::vector<size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");
        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                expval(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    }

    /**
     * @brief Calculate expectation value for a general Observable.
     *
     * @param ob Observable.
     * @return Expectation value with respect to the given observable.
     */
    auto expval(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector);
        ob.applyInPlace(ob_sv);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        auto expect =
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x;
        return static_cast<PrecisionT>(expect);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<ComplexT> &matrix,
                const std::vector<size_t> &wires) -> PrecisionT {
        return this->_statevector.expval(wires, matrix);
    }

    /**
     * @brief Expected value of an observable.
     *
     * @param pauli_words Vector of operators' name strings.
     * @param target_wires Vector of wires where to apply the operator.
     * @param coeffs Complex buffer of size |pauli_words|
     * @return Floating point expected value of the observable.
     */
    auto expval(const std::vector<std::string> &pauli_words,
                const std::vector<std::vector<std::size_t>> &target_wires,
                const std::complex<PrecisionT> *coeffs) -> PrecisionT {
        return this->_statevector.getExpectationValuePauliWords(
            pauli_words, target_wires, coeffs);
    }

    /**
     * @brief Calculate variance of a general Observable.
     *
     * @param ob Observable.
     * @return Variance with respect to the given observable.
     */
    auto var(const Observable<StateVectorT> &ob) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob.applyInPlace(ob_sv);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    }

    /**
     * @brief Variance of an observable.
     *
     * @param operation String with the operator name.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::string &operation, const std::vector<size_t> &wires)
        -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob_sv.applyOperation(operation, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance of an observable.
     *
     * @param matrix Square matrix in row-major order.
     * @param wires Wires where to apply the operator.
     * @return Floating point with the variance of the observable.
     */
    auto var(const std::vector<ComplexT> &matrix,
             const std::vector<size_t> &wires) -> PrecisionT {
        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());
        ob_sv.applyMatrix(matrix, wires);

        auto device_id = ob_sv.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id = ob_sv.getDataBuffer().getDevTag().getStreamID();

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };

    /**
     * @brief Variance for a list of observables.
     *
     * @tparam op_type Operation type.
     * @param operations_list List of operations to measure.
     * Square matrix in row-major order or string with the operator name.
     * @param wires_list List of wires where to apply the operators.
     * @return Floating point std::vector with the variance of the
     observables.
     */
    template <typename op_type>
    auto var(const std::vector<op_type> &operations_list,
             const std::vector<std::vector<size_t>> &wires_list)
        -> std::vector<PrecisionT> {
        PL_ABORT_IF(
            (operations_list.size() != wires_list.size()),
            "The lengths of the list of operations and wires do not match.");

        std::vector<PrecisionT> expected_value_list;

        for (size_t index = 0; index < operations_list.size(); index++) {
            expected_value_list.emplace_back(
                var(operations_list[index], wires_list[index]));
        }

        return expected_value_list;
    };

    /**
     * @brief Variance of a sparse Hamiltonian.
     *
     * @tparam index_type integer type used as indices of the sparse matrix.
     * @param row_map_ptr   row_map array pointer.
     *                      The j element encodes the number of non-zeros
     above
     * row j.
     * @param row_map_size  row_map array size.
     * @param entries_ptr   pointer to an array with column indices of the
     * non-zero elements.
     * @param values_ptr    pointer to an array with the non-zero elements.
     * @param numNNZ        number of non-zero elements.
     * @return Floating point with the variance of the sparse Hamiltonian.
     */
    template <class index_type>
    PrecisionT
    var(const index_type *csrOffsets_ptr, const index_type csrOffsets_size,
        const index_type *columns_ptr,
        const std::complex<PrecisionT> *values_ptr, const index_type numNNZ) {

        PL_ABORT_IF(
            (this->_statevector.getLength() != (size_t(csrOffsets_size) - 1)),
            "Statevector and Hamiltonian have incompatible sizes.");

        StateVectorT ob_sv(this->_statevector.getData(),
                           this->_statevector.getLength());

        auto device_id =
            this->_statevector.getDataBuffer().getDevTag().getDeviceID();
        auto stream_id =
            this->_statevector.getDataBuffer().getDevTag().getStreamID();
        cusparseHandle_t handle = this->_statevector.getCusparseHandle();

        cuUtil::SparseMV_cuSparse<index_type, PrecisionT, CFP_t>(
            csrOffsets_ptr, csrOffsets_size, columns_ptr, values_ptr, numNNZ,
            this->_statevector.getData(), ob_sv.getData(), device_id, stream_id,
            handle);

        const PrecisionT mean_square =
            innerProdC_CUDA(ob_sv.getData(), ob_sv.getData(), ob_sv.getLength(),
                            device_id, stream_id, ob_sv.getCublasCaller())
                .x;

        const PrecisionT squared_mean = static_cast<PrecisionT>(std::pow(
            innerProdC_CUDA(this->_statevector.getData(), ob_sv.getData(),
                            this->_statevector.getLength(), device_id,
                            stream_id, this->_statevector.getCublasCaller())
                .x,
            2));
        return (mean_square - squared_mean);
    };
}; // class Measurements
} // namespace Pennylane::LightningGPU::Measures