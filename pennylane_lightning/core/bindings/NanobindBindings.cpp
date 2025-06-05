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
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Mark STL containers as opaque
NB_MAKE_OPAQUE(std::vector<int>)
// NB_MAKE_OPAQUE(std::map<std::string, std::string>)

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
 * @brief Simple info function that returns a nanobind/python dictionary
 * directly
 */
nb::dict nb_info() {
    nb::dict info;
    info["binding_type"] = "nanobind";
    info["version"] = "1.0.0";
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
}
#endif
