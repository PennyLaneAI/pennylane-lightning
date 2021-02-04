// Copyright 2020 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "StateVector.hpp"

Pennylane::StateVector::StateVector(pybind11::array_t<CplxType>* numpyArray) {
    pybind11::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1)
        throw std::invalid_argument("NumPy array must be a 1-dimensional array");
    if (numpyArrayInfo.itemsize != sizeof(CplxType))
        throw std::invalid_argument("NumPy array must be a complex double-precision array");

    this->arr = (CplxType*)numpyArrayInfo.ptr;
    this->length = numpyArrayInfo.shape[0];
}

Pennylane::StateVector::StateVector(CplxType* arr, size_t length) {
    this->arr = arr;
    this->length = length;
}
