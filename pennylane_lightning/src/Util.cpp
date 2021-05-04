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
#include "Util.hpp"
#include <math.h> 

void Pennylane::set_block(Pennylane::CplxType* mx, const size_t &dim, const size_t &start_ind, CplxType* block_mx, const size_t &block_dim){
    auto row_of_start = floor(start_ind/dim);
    auto col_of_start = start_ind % dim;

    if (dim - row_of_start < block_dim || dim - col_of_start < block_dim)
        throw std::invalid_argument(std::string("The block of the matrix determined by the start index needs to be greater than or equal to the dimension of the submatrix."));

    size_t i = 0;
    for(size_t j = 0; j<block_dim*block_dim; j+=block_dim){
        for(size_t k = 0; k<block_dim; ++k){
            mx[start_ind + i + k] = block_mx[k+j];
        }
        i += dim;
    }
}

void Pennylane::swap_cols(CplxType* mx, const size_t &dim, const size_t column1, const size_t column2){
    if (column1 >= dim || column2 >= dim)
        throw std::invalid_argument(std::string("The indices of the columns need to be smaller than the dimension of the matrix."));

    for(size_t i=0; i<dim; ++i){
        auto row_num = i*dim;
        std::swap(mx[row_num +column1], mx[row_num +column2]);
    }
}

void Pennylane::swap_rows(CplxType* mx, const size_t &dim, const size_t row1, const size_t row2){
    if (row1 >= dim || row2 >= dim)
        throw std::invalid_argument(std::string("The indices of the rows need to be smaller than the dimension of the matrix."));

    for(size_t i = 0; i<dim; ++i){
        std::swap(mx[row1 * dim +i], mx[row2 * dim +i]);
    }
}

std::vector<Pennylane::CplxType> Pennylane::create_identity(const size_t & dim){
    std::vector<CplxType> identity(dim * dim);
    for (size_t i = 0; i< identity.size(); i+=(dim+1)){
        identity.at(i) = 1;
    }
    return identity;
}

