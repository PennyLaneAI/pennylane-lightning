
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

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <iostream>

#include "SharedLibLoader.hpp"

#include "config.h"

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
std::unordered_map<std::string, std::size_t> priority_lib = {
    {"stdc", 0}, {"gcc", 1}, {"quadmath", 2}, {"gfortran", 3}, {"openblas", 4}};
} // namespace
/// @endcond

namespace Pennylane::Util {

// Exclusively for python calls and tested in the python layer
// LCOV_EXCL_START
#ifdef __linux__
/**
 * @brief Get the path to the current shared library object.
 *
 * @return const char*
 */
inline const char *getPath() {
    Dl_info dl_info;
    PL_ABORT_IF(dladdr((const void *)getPath, &dl_info) == 0,
                "Can't get the path to the shared library.");
    return dl_info.dli_fname;
}
// TODO add windows support
// #elif defined(_MSC_VER)
// inline std::string getPath() {
//     char buffer[MAX_PATH];
//     GetModuleFileName(nullptr, buffer, MAX_PATH);
//     std::string fullPath(buffer);
//     std::size_t pos = fullPath.find_last_of("\\/");
//     return fullPath.substr(0, pos);
// }
#endif
// LCOV_EXCL_STOP

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
    const std::string libName(SCIPY_LIBS_PATH);
    std::shared_ptr<SharedLibLoader> blasLib =
        std::make_shared<SharedLibLoader>(libName);
    // LCOV_EXCL_STOP
#else
    std::shared_ptr<SharedLibLoader> blasLib;
    std::vector<std::shared_ptr<SharedLibLoader>> blasLibs;
    // For C++ usage
    std::string scipyPathStr(SCIPY_LIBS_PATH);

    // Exclusively for python calls
    // LCOV_EXCL_START
    if (!std::filesystem::exists(scipyPathStr)) {
        std::string currentPathStr(getPath());
        std::string site_packages_str("site-packages/");

        std::size_t str_pos = currentPathStr.find(site_packages_str);
        if (str_pos != std::string::npos) {
            scipyPathStr =
                currentPathStr.substr(0, str_pos + site_packages_str.size());
            scipyPathStr += "scipy.libs";
        }

        try {
            // convert the relative path to absolute path
            scipyPathStr = std::filesystem::canonical(scipyPathStr).string();
        } catch (const std::exception &err) {
            std::cout << "Canonical path for scipy.libs"
                      << " threw exception:\n"
                      << err.what() << '\n';
        }
    }
    // LCOV_EXCL_STOP

    std::filesystem::path scipyLibsPath(scipyPathStr);

    std::vector<std::pair<std::string, std::size_t>> availableLibs;

    for (const auto &lib : std::filesystem::directory_iterator(scipyLibsPath)) {
        if (lib.is_regular_file()) {
            for (const auto &iter : priority_lib) {
                std::string libname_str = lib.path().filename().string();
                if (libname_str.find(iter.first) != std::string ::npos) {
                    availableLibs.emplace_back(libname_str, iter.second);
                }
            }
        }
    }

    std::sort(availableLibs.begin(), availableLibs.end(),
              [](const auto &lhs, const auto &rhs) {
                  return lhs.second < rhs.second;
              });

    for (const auto &lib : availableLibs) {
        auto libPath = scipyLibsPath / lib.first.c_str();
        const std::string libPathStr = libPath.string();
        blasLibs.emplace_back(std::make_shared<SharedLibLoader>(libPathStr));
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
