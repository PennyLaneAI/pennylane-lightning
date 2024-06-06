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
 * @file LightningKokkos.hpp
 */

#pragma once

#include "LightningBase.hpp"
#include <Kokkos_Core.hpp>

namespace Pennylane::LightningKokkos {
/**
 * @brief  Kokkos state vector class
 *
 * @tparam fp_t Floating-point precision type.
 */
template <class fp_t = double>
class StateVectorKokkos final
    : public StateVectorBase<fp_t, StateVectorKokkos<fp_t>> {
  public:
    using PrecisionT = fp_t;
    using ComplexT = Kokkos::complex<fp_t>;
    using KokkosVector = Kokkos::View<ComplexT *>;

    StateVectorKokkos();
    StateVectorKokkos(std::size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    /**
     * @brief Init zeros for the state-vector on device.
     */
    void initZeros();

    /**
     * @brief Set value for a single element of the state-vector on device.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const std::size_t index);

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values);

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     * @param num_qubits Number of qubits
     */
    void resetStateVector();

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    StateVectorKokkos(std::complex<PrecisionT> *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(const ComplexT *hostdata_, std::size_t length,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    /**
     * @brief Create a new state vector from data on the host.
     *
     * @param num_qubits Number of qubits
     */
    StateVectorKokkos(std::vector<ComplexT> hostdata_,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    /**
     * @brief Copy constructor
     *
     * @param other Another state vector
     */
    StateVectorKokkos(const StateVectorKokkos &other,
                      const Kokkos::InitializationSettings &kokkos_args = {});

    /**
     * @brief Destructor for StateVectorKokkos class
     *
     * @param other Another state vector
     */
    ~StateVectorKokkos();

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {});

    template <bool inverse = false>
    void applyControlledGlobalPhase(const std::vector<ComplexT> &diagonal);

    /**
     * @brief Apply a single gate to the state vector.
     *
     * @param opName Name of gate to apply.
     * @param controlled_wires Control wires.
     * @param controlled_values Control values (false or true).
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     * @param gate_matrix Optional std gate matrix if opName doesn't exist.
     */
    void applyOperation(const std::string &opName,
                        const std::vector<std::size_t> &controlled_wires,
                        const std::vector<bool> &controlled_values,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {},
                        const std::vector<ComplexT> &gate_matrix = {});
    /**
     * @brief Apply a multi qubit operator to the state vector using a matrix
     *
     * @param matrix Kokkos gate matrix in the device space
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     */
    void applyMultiQubitOp(const KokkosVector matrix,
                           const std::vector<std::size_t> &wires,
                           bool inverse = false);

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false);

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(std::vector<ComplexT> &matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false);

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false);

    /**
     * @brief Apply a given matrix directly to the statevector.
     *
     * @param matrix Matrix data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const std::vector<ComplexT> &matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false);

    /**
     * @brief Apply a single generator to the state vector using the given
     * kernel.
     *
     * @param opName Name of gate to apply.
     * @param wires Wires to apply gate to.
     * @param inverse Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    auto applyGenerator(const std::string &opName,
                        const std::vector<std::size_t> &wires,
                        bool inverse = false,
                        const std::vector<fp_t> &params = {}) -> fp_t;

    /**
     * @brief Collapse the state vector after having measured one of the
     * qubits.
     *
     * The branch parameter imposes the measurement result on the given wire.
     *
     * @param wire Wire to collapse.
     * @param branch Branch 0 or 1.
     */
    void collapse(std::size_t wire, bool branch);

    /**
     * @brief Normalize vector (to have norm 1).
     */
    void normalize();

    /**
     * @brief Update data of the class
     *
     * @param other Kokkos View
     */
    void updateData(const KokkosVector other);

    /**
     * @brief Update data of the class
     *
     * @param other State vector
     */
    void updateData(const StateVectorKokkos<fp_t> &other);

    /**
     * @brief Update data of the class
     *
     * @param new_data data pointer to new data.
     * @param new_size size of underlying data storage.
     */
    void updateData(ComplexT *new_data, std::size_t new_size);

    /**
     * @brief Update data of the class
     *
     * @param other STL vector of type ComplexT
     */
    void updateData(std::vector<ComplexT> &other);

    [[nodiscard]] auto getData() -> ComplexT *;

    [[nodiscard]] auto getData() const -> const ComplexT *;

    /**
     * @brief Get the Kokkos data of the state vector.
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() const -> KokkosVector;

    /**
     * @brief Get the Kokkos data of the state vector
     *
     * @return The pointer to the data of state vector
     */
    [[nodiscard]] auto getView() -> KokkosVector;

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector() -> std::vector<ComplexT>;

    [[nodiscard]] auto getDataVector() const -> const std::vector<ComplexT>;

    /**
     * @brief Copy data from the host space to the device space.
     *
     */
    inline void HostToDevice(ComplexT *sv, std::size_t length);

    /**
     * @brief Copy data from the device space to the host space.
     *
     */
    inline void DeviceToHost(ComplexT *sv, std::size_t length) const;

    /**
     * @brief Copy data from the device space to the device space.
     *
     */
    inline void DeviceToDevice(KokkosVector vector_to_copy);
};

}; // namespace Pennylane::LightningKokkos
