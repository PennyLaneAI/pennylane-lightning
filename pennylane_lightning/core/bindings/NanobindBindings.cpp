// Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
 * @file NanobindBindings.cpp
 * Export C++ functions to Python using Nanobind.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <complex>
#include <map>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Mark STL containers as opaque
NB_MAKE_OPAQUE(std::vector<int>)
NB_MAKE_OPAQUE(std::vector<double>)

// Defining the module name with _nb suffix to differentiate from pybind11
// modules
#if defined(_ENABLE_PLQUBIT)
#define LIGHTNING_NB_MODULE_NAME lightning_qubit_nb
#elif _ENABLE_PLKOKKOS == 1
#define LIGHTNING_NB_MODULE_NAME lightning_kokkos_nb
#elif _ENABLE_PLGPU == 1
#define LIGHTNING_NB_MODULE_NAME lightning_gpu_nb
#elif _ENABLE_PLTENSOR == 1
#define LIGHTNING_NB_MODULE_NAME lightning_tensor_nb
#endif

/**
 * @brief Simple info function that returns a nanobind dictionary directly
 */
nb::dict nb_info() {
    nb::dict info;
    info["binding_type"] = "nanobind";
    info["description"] = "Nanobind implementation of PennyLane-Lightning";
    return info;
}

/**
 * @brief NumPy array version of vector addition
 */
template <typename PrecisionT>
nb::ndarray<nb::numpy, PrecisionT>
nb_add_vectors(nb::ndarray<nb::numpy, const PrecisionT> a,
               nb::ndarray<nb::numpy, const PrecisionT> b) {
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::runtime_error("Arrays must be 1-dimensional");
    }

    if (a.shape(0) != b.shape(0)) {
        throw std::runtime_error("Array sizes must match");
    }

    size_t n = a.shape(0);

    // Allocate memory
    PrecisionT *result_data = new PrecisionT[n];

    // Get input data pointers
    const PrecisionT *a_ptr = a.data();
    const PrecisionT *b_ptr = b.data();

    // Perform addition
    for (size_t i = 0; i < n; ++i) {
        result_data[i] = a_ptr[i] + b_ptr[i];
    }

    // Create ndarray with custom deleter
    size_t shape[1] = {n};
    auto capsule = nb::capsule(result_data, [](void *p) noexcept {
        delete[] static_cast<PrecisionT *>(p);
    });

    return nb::ndarray<nb::numpy, PrecisionT>(result_data, 1, shape, capsule);
}

// Example StateVector class for testing
class SimpleStateVector {
  public:
    using PrecisionT = double;
    using ComplexT = std::complex<PrecisionT>;

    SimpleStateVector(size_t num_qubits) : num_qubits_(num_qubits) {}

    void applyMatrix(const ComplexT* matrix, const std::vector<std::size_t> &wires, bool inverse) {
        // Placeholder implementation that uses parameters to avoid warnings
        (void)matrix;      // Silence unused parameter warning
        (void)wires;       // Silence unused parameter warning
        (void)inverse;     // Silence unused parameter warning
    }

    void applyOperation(const std::string& gate_name, const std::vector<std::size_t> &wires, 
                        bool inverse, const std::vector<PrecisionT> &params) {
        // Placeholder implementation that uses parameters to avoid warnings
        (void)gate_name;   // Silence unused parameter warning
        (void)wires;       // Silence unused parameter warning
        (void)inverse;     // Silence unused parameter warning
        (void)params;      // Silence unused parameter warning
    }

  private:
    size_t num_qubits_;
};

#if defined(LIGHTNING_NB_MODULE_NAME)
/**
 * @brief Add Lightning State-vector C++ classes, methods and functions to
 * Python module using Nanobind.
 */
NB_MODULE(LIGHTNING_NB_MODULE_NAME, m) {
    // Register basic info function
    m.def("nb_info", &nb_info,
          "Get information about the Nanobind implementation");

    // Register NumPy array operation
    m.def("nb_add_vectors", &nb_add_vectors<double>, "a"_a, "b"_a,
          "Add two NumPy arrays element-wise");

    // Add a simple class
    nb::class_<std::vector<int>>(m, "IntVector")
        .def(nb::init<>())
        .def("append", [](std::vector<int> &v, int x) { v.push_back(x); })
        .def("__len__", [](const std::vector<int> &v) { return v.size(); })
        .def("__getitem__", [](const std::vector<int> &v, size_t i) {
            if (i >= v.size())
                throw nb::index_error();
            return v[i];
        });

    // Test the SimpleStateVector class
    nb::class_<SimpleStateVector> sv_class(m, "SimpleStateVector");
    sv_class.def(nb::init<size_t>());

    // Manually register a few methods
    sv_class.def(
        "applyMatrix",
        [](SimpleStateVector &sv,
           const nb::ndarray<std::complex<double>, nb::numpy> &matrix,
           const std::vector<std::size_t> &wires, bool inverse) {
            const std::complex<double> *data_ptr =
                reinterpret_cast<const std::complex<double> *>(matrix.data());
            sv.applyMatrix(data_ptr, wires, inverse);
        },
        "Apply a given matrix to wires.");
}
#endif
