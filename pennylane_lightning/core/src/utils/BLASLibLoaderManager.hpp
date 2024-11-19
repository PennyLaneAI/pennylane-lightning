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
 * LAPACK/BLAS Lib dynamic loader manager.
 */
#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <iostream>

#include "SharedLibLoader.hpp"

namespace {
// Exclusively for python calls and tested in the python layer
// LCOV_EXCL_START
#ifndef _MSC_VER
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
#elif defined(_MSC_VER)
inline std::string getPath() {
    char buffer[MAX_PATH];
    GetModuleFileName(nullptr, buffer, MAX_PATH);
    std::string fullPath(buffer);
    std::size_t pos = fullPath.find_last_of("\\/");
    return fullPath.substr(0, pos);
}
#endif
// LCOV_EXCL_STOP
} // namespace

namespace Pennylane::Util {
/**
 * @brief BLAS Lib dynamic loader manager.
 *
 * This class is a singleton that manages the dynamic loading of BLAS libraries.
 * It will search for the BLAS libraries in the given path, or in the default
 * locations if no path is provided. The default locations are:
 * - The path provided by the SCIPY_OPENBLAS32_LIB macro.
 * - The path provided by the get_scipylibs_path() function.
 *
 * The class will search for the libraries in the following order:
 * - openblas
 *
 * The class will load the first library found in the order above.
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

    static std::string get_scipylibs_path_worker_() {
        if (std::filesystem::exists(SCIPY_OPENBLAS32_LIB)) {
            return SCIPY_OPENBLAS32_LIB;
        }

        std::string scipyPathStr;
        std::string currentPathStr(getPath());

#ifdef _MSC_VER
        std::string site_packages_str("site-packages\\");
#else
        std::string site_packages_str("site-packages/");
#endif

        std::size_t str_pos = currentPathStr.find(site_packages_str);
        if (str_pos != std::string::npos) {
            scipyPathStr =
                currentPathStr.substr(0, str_pos + site_packages_str.size());
#ifdef _MSC_VER
            scipyPathStr += "scipy_openblas32\\lib";
#else
            scipyPathStr += "scipy_openblas32/lib";
#endif
        }

        if (std::filesystem::exists(scipyPathStr)) {
            try {
                // convert the relative path to absolute path
                scipyPathStr =
                    std::filesystem::canonical(scipyPathStr).string();
            } catch (const std::exception &err) {
                std::cerr << "Canonical path for scipy_openblas32"
                          << " threw exception:\n"
                          << err.what() << '\n';
            }
        } else {
            try {
#ifdef _MSC_VER
                scipyPathStr = currentPathStr + "..\\..\\scipy_openblas32\\lib";
#else
                scipyPathStr = currentPathStr + "../../scipy_openblas32/lib";
#endif
                // convert the relative path to absolute path
                scipyPathStr =
                    std::filesystem::canonical(scipyPathStr).string();
            } catch (const std::exception &err) {
                std::cerr << "Canonical path for scipy_openblas32"
                          << " threw exception:\n"
                          << err.what() << '\n';
            }
        }

        return scipyPathStr;
    }
    /**
     * @brief Get the path to the scipy_openblas32/lib package.
     *
     * This function will return the path to the scipy_openblas32/lib
     * package. It will first try to get the path from the current Python
     * environment. This method only works for Python layer calls, which
     * means a Python interpreter is running.
     *
     * @return std::string The path to the scipy_openblas32/lib package.
     */
    static std::string get_scipylibs_path_() {
        return get_scipylibs_path_worker_();
    }
    /**
     * @brief BLASLibLoaderManager.
     *
     * This function will initialize the BLASLibLoaderManager by searching
     * for the BLAS libraries in the given path.
     *
     * @param blaslib_path The path to the BLAS libraries.
     */
    explicit BLASLibLoaderManager() {
        std::string scipyPathStr = get_scipylibs_path_();
        std::filesystem::path scipyLibsPath(scipyPathStr);
        auto libPath = scipyLibsPath / blas_lib_name_.c_str();
        blasLib_ = std::make_shared<SharedLibLoader>(libPath.string());
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