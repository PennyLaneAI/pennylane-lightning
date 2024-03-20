
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
 * [@file](https://github.com/file)
 * LAPACK wrapper functions declarations.
 */
#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#elif defined(_MSC_VER)
#include <cstdlib>
#include <filesystem>
#include <windows.h>
#endif

/// @cond DEV
namespace {
// LAPACK routine for complex Hermitian eigensystems
typedef void (*zheevPtr)(const char *, const char *, const int *,
                         std::complex<double> *, const int *, double *,
                         std::complex<double> *, const int *, double *, int *);
typedef void (*cheevPtr)(const char *, const char *, const int *,
                         std::complex<float> *, const int *, float *,
                         std::complex<float> *, const int *, float *, int *);

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
    void *handle =
        dlopen("/System/Library/Frameworks/Accelerate.framework/Versions/"
               "Current/Frameworks/vecLib.framework/libLAPACK.dylib",
               RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        handle = dlopen("/usr/local/opt/lapack/lib/liblapack.dylib",
                        RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
    }
#elif defined(__linux__)
    void *handle = dlopen("liblapack.so", RTLD_LAZY | RTLD_GLOBAL);
#elif defined(_MSC_VER)
    const char *PythonSitePackagePath = std::getenv("PYTHON_SITE_PACKAGES");
    std::string openblasLib;
    if (PythonSitePackagePath != nullptr) {
        std::filesystem::path scipyLibsPath(PythonSitePackagePath);
        scipyLibsPath = scipyLibsPath / "scipy.libs";
        for (const auto &lib :
             std::filesystem::directory_iterator(scipyLibsPath)) {
            if (lib.is_regular_file()) {
                if (lib.path().filename().find("openblas") !=
                    std::string::npos) {
                    openblasLib = lib.path().filename().c_str();
                }
            }
        }
    } else {
        auto currentPath = std::filesystem::currentPath();
        auto scipyLibsPath = currentPath.parent_path() / "scipy.libs";
        for (const auto &lib :
             std::filesystem::directory_iterator(scipyLibsPath)) {
            if (lib.is_regular_file()) {
                if (lib.path().filename().find("openblas") !=
                    std::string::npos) {
                    openblasLib = lib.path().filename().c_str();
                }
            }
        }
    }
    HMODULE handle = LoadLibrary(openblasLib.c_str());
#endif

    char jobz = 'V'; // Enable both eigenvalues and eigenvectors computation
    char uplo = 'L'; // Upper triangle of matrix is stored
    std::vector<std::complex<T>> work_query(1); // Vector for optimal size query
    int lwork = -1;                             // Optimal workspace size query
    std::vector<T> rwork(3 * n - 2);            // Real workspace array
    int info;

    if constexpr (std::is_same<T, float>::value) {
#if defined(__APPLE__) || defined(__linux__)
        cheevPtr cheev = reinterpret_cast<cheevPtr>(dlsym(handle, "cheev_"));
#elif defined(_MSC_VER)
        cheevPtr cheev =
            reinterpret_cast<cheevPtr>(GetProcAddress(handle, "cheev_"));
#endif
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
#if defined(__APPLE__) || defined(__linux__)
        zheevPtr zheev = reinterpret_cast<zheevPtr>(dlsym(handle, "zheev_"));
#elif defined(_MSC_VER)
        zheevPtr zheev =
            reinterpret_cast<zheevPtr>(GetProcAddress(handle, "zheev_"));
#endif
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

#if defined(__APPLE__) || defined(__linux__)
    dlclose(handle);
#elif defined(_MSC_VER)
    FreeLibrary(handle);
#endif

    std::transform(ah.begin(), ah.end(), unitary.begin(),
                   [](std::complex<T> value) {
                       return std::complex<T>{value.real(), -value.imag()};
                   });
}
} // namespace Pennylane::Util
