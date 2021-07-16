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
#include "Apply.hpp"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

using Pennylane::CplxType;
using Pennylane::StateVector;
using std::string;
using std::vector;

template <class Precision = double>
static StateVector
create(const pybind11::array_t<std::complex<Precision>> *numpyArray) {
    pybind11::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1)
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    if (numpyArrayInfo.itemsize != sizeof(std::complex<Precision>))
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    const std::complex<Precision> *data_ptr =
        static_cast<std::complex<Precision> *>(buf1.ptr);
    return StateVector<Precision>({data_ptr, data_ptr + buf1.shape[0]});
}

template <class Precision = double>
void apply(pybind11::array_t<std::complex<Precision>> &stateNumpyArray,
           vector<string> ops, vector<vector<unsigned int>> wires,
           vector<vector<Precision>> params, vector<bool> inverse,
           const unsigned int qubits) {
    StateVector<Precision> state = create(&stateNumpyArray);
    Pennylane::apply(state, ops, wires, params, inverse, qubits);
}

PYBIND11_MODULE(lightning_qubit_ops, m) {
    m.doc() = "lightning.qubit apply() method";
    m.def("apply", apply<double>, "lightning.qubit apply() method");
    m.def("apply", apply<float>, "lightning.qubit apply() method");
}
