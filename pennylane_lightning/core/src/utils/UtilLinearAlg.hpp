
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
 * @file
 * LAPACK wrapper functions declarations.
 */
#pragma once

#include <Python.h>
#include <algorithm>
#include <array>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include <iostream>

#include "SharedLibLoader.hpp"

/// @cond DEV
namespace {
// Declare heev function pointers to access corresponding functions in
// LAPACK/OpenBLAS
using zheevPtr = void (*)(const char *, const char *, const int *,
                          std::complex<double> *, const int *, double *,
                          std::complex<double> *, const int *, double *, int *);
using cheevPtr = void (*)(const char *, const char *, const int *,
                          std::complex<float> *, const int *, float *,
                          std::complex<float> *, const int *, float *, int *);

// Priority table used to sort openblas and its dependencies
std::array<std::string, 5> priority_lib{"stdc", "gcc.", "quadmath", "gfortran",
                                        "openblas"};

/*
std::string get_scipylibs_path_worker() {
    pybind11::object avail_site_packages =
        pybind11::module::import("site").attr("getsitepackages")();

    std::string scipy_lib_path;

    for (auto item : avail_site_packages) {
        std::string tmp_path = pybind11::str(item);
        tmp_path += "/scipy.libs";
        if (std::filesystem::exists(tmp_path)) {
            return tmp_path;
        }
    }

    PL_ABORT_IF(scipy_lib_path.empty(), "Can't find scipy.libs");

    return scipy_lib_path;
}

std::string get_scipylibs_path() {
    if (Py_IsInitialized()) {
        return get_scipylibs_path_worker();
    }

    pybind11::scoped_interpreter scope_guard{};
    return get_scipylibs_path_worker();
}
*/

std::string get_scipylibs_path_worker() {
    PyObject* scipy_module = PyImport_ImportModule("scipy");

    PL_ABORT_IF(!scipy_module, "Can't find the scipy module");

    PyObject* scipy_path = PyObject_GetAttrString(scipy_module, "__path__");
    
    std::string scipy_lib_path;

    if (PyList_Check(scipy_path)) {
        // Iterate over the elements in the list
        for (Py_ssize_t i = 0; i < PyList_Size(scipy_path); i++) {
            PyObject* path_item = PyList_GetItem(scipy_path, i);
            std::string path_str(PyUnicode_AsUTF8(path_item));
            std::filesystem::path path2scipy =  path_str;
            std::filesystem::path path2scipylibs = path2scipy/ ".."/"scipy.libs";
            std::filesystem::path absolute_scipylibs = std::filesystem::canonical(path2scipylibs);
            if (std::filesystem::exists(absolute_scipylibs)) {
                Py_DECREF(scipy_path);
                Py_DECREF(scipy_module);
                return absolute_scipylibs.string();
            }
        }
    } else {
        PyObject* path_item = PySequence_GetItem(scipy_path, 0);
        std::string path_str(PyUnicode_AsUTF8(path_item));
        std::filesystem::path path2scipy =  path_str;
        std::filesystem::path path2scipylibs = path2scipy/ ".."/"scipy.libs";
        std::filesystem::path absolute_scipylibs = std::filesystem::canonical(path2scipylibs);

        if (std::filesystem::exists(absolute_scipylibs)) {
            Py_DECREF(scipy_path);
            Py_DECREF(scipy_module);
            return absolute_scipylibs.string();
        }
            
    }

    // Cleanup
    Py_DECREF(scipy_path);
    Py_DECREF(scipy_module);


    return scipy_lib_path;
}

std::string get_scipylibs_path() {
    if (Py_IsInitialized()) {
        return get_scipylibs_path_worker();
    } else {
        Py_Initialize();
        auto pathStr = get_scipylibs_path_worker();
        Py_Finalize();
        return pathStr;
    }
}

} // namespace
/// @endcond

namespace Pennylane::Util {

/**
 * @brief Decompose Hermitian matrix into diagonal matrix and unitaries
 *
 * @tparam T Data type.
 *
 * @param n Number of columns.
 * @param lda Number of rows.
 * @param Ah Hermitian matrix to be decomposed.
 * @param eigenVals eigenvalue results.
 * @param unitaries unitary result.
 */

template <typename T>
void compute_diagonalizing_gates(int n, int lda,
                                 const std::vector<std::complex<T>> &Ah,
                                 std::vector<T> &eigenVals,
                                 std::vector<std::complex<T>> &unitary) {
    eigenVals.clear();
    eigenVals.resize(n);
    unitary = std::vector<std::complex<T>>(n * n, {0, 0});

    std::vector<std::complex<T>> ah(n * lda, {0.0, 0.0});

    // TODO optmize transpose
    for (size_t i = 0; i < static_cast<size_t>(n); i++) {
        for (size_t j = 0; j <= i; j++) {
            ah[j * n + i] = Ah[i * lda + j];
        }
    }
#ifdef __APPLE__
    // LCOV_EXCL_START
    const std::string libName =
        "/System/Library/Frameworks/Accelerate.framework/Versions/Current/"
        "Frameworks/vecLib.framework/libLAPACK.dylib";
    std::shared_ptr<SharedLibLoader> blasLib =
        std::make_shared<SharedLibLoader>(libName);
    // LCOV_EXCL_STOP
#else
    std::shared_ptr<SharedLibLoader> blasLib;
    std::vector<std::shared_ptr<SharedLibLoader>> blasLibs;

    std::filesystem::path scipyLibsPath(get_scipylibs_path());

    std::vector<std::string> availableLibs;
    availableLibs.reserve(priority_lib.size());

    for (const auto &iter : priority_lib) {
        for (const auto &lib :
             std::filesystem::directory_iterator(scipyLibsPath)) {
            if (lib.is_regular_file()) {
                std::string libname_str = lib.path().filename().string();
                if (libname_str.find(iter) != std::string::npos) {
                    availableLibs.push_back(libname_str);
                }
            }
        }
    }

    for (const auto &lib : availableLibs) {
        auto libPath = scipyLibsPath / lib.c_str();
        blasLibs.emplace_back(
            std::make_shared<SharedLibLoader>(libPath.string()));
    }

    blasLib = blasLibs.back();
#endif

    char jobz = 'V'; // Enable both eigenvalues and eigenvectors computation
    char uplo = 'L'; // Upper triangle of matrix is stored
    std::vector<std::complex<T>> work_query(1); // Vector for optimal size query
    int lwork = -1;                             // Optimal workspace size query
    std::vector<T> rwork(3 * n - 2);            // Real workspace array
    int info;

    if constexpr (std::is_same<T, float>::value) {
        cheevPtr cheev =
            reinterpret_cast<cheevPtr>(blasLib->getSymbol("cheev_"));
        // Query optimal workspace size
        cheev(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
              work_query.data(), &lwork, rwork.data(), &info);
        // Allocate workspace
        lwork = static_cast<int>(work_query[0].real());
        std::vector<std::complex<T>> work_optimal(lwork, {0, 0});
        // Perform eigenvalue and eigenvector computation
        cheev(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
              work_optimal.data(), &lwork, rwork.data(), &info);
    } else {
        zheevPtr zheev =
            reinterpret_cast<zheevPtr>(blasLib->getSymbol("zheev_"));
        // Query optimal workspace size
        zheev(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
              work_query.data(), &lwork, rwork.data(), &info);
        // Allocate workspace
        lwork = static_cast<int>(work_query[0].real());
        std::vector<std::complex<T>> work_optimal(lwork, {0, 0});
        // Perform eigenvalue and eigenvector computation
        zheev(&jobz, &uplo, &n, ah.data(), &lda, eigenVals.data(),
              work_optimal.data(), &lwork, rwork.data(), &info);
    }

    std::transform(ah.begin(), ah.end(), unitary.begin(),
                   [](std::complex<T> value) {
                       return std::complex<T>{value.real(), -value.imag()};
                   });
}
} // namespace Pennylane::Util
