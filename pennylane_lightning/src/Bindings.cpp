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
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

using Pennylane::StateVector;
using std::complex;
using std::string;
using std::vector;

template <class T = double>
static StateVector<T> create(const pybind11::array_t<complex<T>> *numpyArray) {
    pybind11::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1)
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    if (numpyArrayInfo.itemsize != sizeof(complex<T>))
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    complex<T> *data_ptr = static_cast<complex<T> *>(numpyArrayInfo.ptr);
    return StateVector<T>({data_ptr, numpyArrayInfo.shape[0]});
}

template <class T = double>
void apply(pybind11::array_t<complex<T>> &stateNumpyArray, const vector<string> &ops,
           const vector<vector<size_t>> &wires, const vector<bool> &inverse,
           const vector<vector<T>> &params ) {
    auto state = create<T>(&stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

PYBIND11_MODULE(lightning_qubit_ops, m) {
    m.doc() = "lightning.qubit apply() method";
    m.def("apply", apply<double>, "lightning.qubit apply() method");
    m.def("apply", apply<float>, "lightning.qubit apply() method");
}
