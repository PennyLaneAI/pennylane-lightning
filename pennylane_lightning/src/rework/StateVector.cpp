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
#include "StateVector.hpp"

Pennylane::StateVector Pennylane::StateVector::create(const pybind11::array_t<CplxType>* numpyArray) {
    pybind11::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1)
        throw std::invalid_argument("NumPy array must be a 1-dimensional array");
    if (numpyArrayInfo.itemsize != sizeof(CplxType))
        throw std::invalid_argument("NumPy array must be a complex128 array");

    return StateVector((CplxType*)numpyArrayInfo.ptr, numpyArrayInfo.shape[0]);
}

Pennylane::StateVector::StateVector(CplxType* arr, size_t length)
    : arr(arr)
    , length(length)
{}
