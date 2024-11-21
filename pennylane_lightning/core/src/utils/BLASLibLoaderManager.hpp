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
 * BLAS Lib dynamic loader manager.
 */
#pragma once

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#ifndef SCIPY_OPENBLAS32_LIB
#define SCIPY_OPENBLAS32_LIB ""
#endif

#include "SharedLibLoader.hpp"

namespace Pennylane::Util {
/**
 * @brief BLAS Lib dynamic loader manager.
 *
 * This class is a singleton that manages the dynamic loading of BLAS libraries.
 * It will search for the BLAS libraries in the given path, or in the RPATH
 * - The path provided by the SCIPY_OPENBLAS32_LIB macro.
 */
class BLASLibLoaderManager {
  private:
#ifdef __APPLE__
    const std::string blas_lib_name_ = "libscipy_openblas.dylib";
#elif defined(_MSC_VER)
    const std::string blas_lib_name_ = "libscipy_openblas.dll";
#else
    const std::string blas_lib_name_ = "libscipy_openblas.so";
#endif

    std::shared_ptr<SharedLibLoader> blasLib_;
    const std::string get_scipylibs_path_worker_() {
        // Pure C++ lib support only, static libs solution
        if (std::filesystem::exists(SCIPY_OPENBLAS32_LIB)) {
            std::filesystem::path scipyLibsPath(SCIPY_OPENBLAS32_LIB);
            auto libPath = scipyLibsPath / blas_lib_name_.c_str();
            return libPath.string();
        }

        // The following code is only for Python layer calls, shared libs
        // solution LCOV_EXCL_START
        return blas_lib_name_;
        // LCOV_EXCL_STOP
    };
    /**
     * @brief BLASLibLoaderManager.
     */
    explicit BLASLibLoaderManager() {
        blasLib_ =
            std::make_shared<SharedLibLoader>(get_scipylibs_path_worker_());
    }

  public:
    BLASLibLoaderManager(BLASLibLoaderManager &&) = delete;
    BLASLibLoaderManager(const BLASLibLoaderManager &) = delete;
    BLASLibLoaderManager &operator=(const BLASLibLoaderManager &) = delete;
    BLASLibLoaderManager operator=(const BLASLibLoaderManager &&) = delete;

    static BLASLibLoaderManager &getInstance() {
        static BLASLibLoaderManager instance;
        return instance;
    }

    ~BLASLibLoaderManager() = default;

    /**
     * @brief Get the BLAS library.
     *
     * This function will return the BLAS library.
     *
     * @return SharedLibLoader* The BLAS library.
     */
    auto getBLASLib() -> SharedLibLoader * { return blasLib_.get(); }
};
} // namespace Pennylane::Util
