// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * Contains uncategorised utility functions.
 */
#pragma once

#include <assert.h>
#include <memory>
#include <vector>

#include "typedefs.hpp"

namespace Pennylane {

    /**
     * Calculates 2^n for some integer n > 0 using bitshifts.
     *
     * @param n the exponent
     * @return value of 2^n
     */
    inline size_t exp2(const unsigned int& n) {
        return (size_t)1 << n;
    }

    /**
     * Calculates the decimal value for a qubit, assuming a big-endian convention.
     *
     * @param qubitIndex the index of the qubit in the range [0, qubits)
     * @param qubits the number of qubits in the circuit
     * @return decimal value for the qubit at specified index
     */
    inline size_t maxDecimalForQubit(const unsigned int qubitIndex, const unsigned int qubits) {
        assert(qubitIndex < qubits);
        return exp2(qubits - qubitIndex - 1);
    }

    /**
     * Creates the identity matrix with a specific dimension.
     *
     * @param dim the dimension of the matrix
     * @return the identity matrix of size dim
     */
    std::vector<CplxType> create_identity(const size_t & dim);

    /**
     * Sets a matrix block with the defined submatrix in-place.
     *
     * @param mx pointer to the complex-typed matrix in a row-major vector representation
     * @param dim the dimension of the matrix
     * @param start_ind the index from which we set the matrix block
     * @param block_mx the submatrix to use when setting the block
     * @param block_dim the dimension of the submatrix
     */
    void set_block(Pennylane::CplxType* mx, const size_t &dim, const size_t &start_ind, CplxType* block_mx, const size_t &block_dim);

    /**
     * In-place swaps two columns of a matrix.
     *
     * @param mx pointer to the complex-typed matrix in a row-major vector representation
     * @param dim the dimension of the matrix
     * @param column1 the index of the first column
     * @param column2 the index of the second column
     */
    void swap_cols(CplxType* mx, const size_t &dim, const size_t column1, const size_t column2);

    /**
     * In-place swaps two rows of a matrix.
     *
     * @param mx pointer to the complex-typed matrix in a row-major vector representation
     * @param dim the dimension of the matrix
     * @param row1 the index of the first row
     * @param row2 the index of the second row
     */
    void swap_rows(CplxType* mx, const size_t &dim, const size_t row1, const size_t row2);
}

// Helper similar to std::make_unique from c++14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Exception for functions that aren't implemented
class NotImplementedException : public std::logic_error
{
public:
    NotImplementedException() : std::logic_error("Function is not implemented.") { };
};
