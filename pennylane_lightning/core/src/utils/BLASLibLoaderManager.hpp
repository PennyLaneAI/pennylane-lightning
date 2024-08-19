
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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#if not defined(__APPLE__) && not defined(_ENABLE_PYTHON)
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#endif

#include "SharedLibLoader.hpp"

namespace {
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
#elif defined(_MSC_VER)
inline std::string getPath() {
    char buffer[MAX_PATH];
    GetModuleFileName(nullptr, buffer, MAX_PATH);
    std::string fullPath(buffer);
    std::size_t pos = fullPath.find_last_of("\\/");
    return fullPath.substr(0, pos);
}
#endif
} // namespace

namespace Pennylane::Util {
/**
 * @brief BLAS Lib dynamic loader manager.
 *
 * This class is a singleton that manages the dynamic loading of BLAS libraries.
 * It will search for the BLAS libraries in the given path, or in the default
 * locations if no path is provided. The default locations are:
 * - The path provided by the SCIPY_LIBS_PATH macro.
 * - The path provided by the get_scipylibs_path() function.
 * - The path provided by the scipy.libs package.
 * - The path provided by the scipy.libs package in the site-packages directory.
 * - The path provided by the scipy.libs package in the site-packages directory
 * of the current Python environment.
 *
 * The class will search for the libraries in the following order:
 * - stdc
 * - gcc.
 * - quadmath
 * - gfortran
 * - openblas
 *
 * The class will load the first library found in the order above.
 */
class BLASLibLoaderManager {
  private:
    static inline std::array<std::string, 5> priority_lib{
        "stdc", "gcc.", "quadmath", "gfortran", "openblas"};
    bool scipy_prefix_ = false;
    std::vector<std::shared_ptr<SharedLibLoader>> blasLibs;
    std::shared_ptr<SharedLibLoader> blasLib;
#ifdef __APPLE__
    const std::string scipy_lib_path_macos_str =
        "/System/Library/Frameworks/Accelerate.framework/Versions/Current/"
        "Frameworks/vecLib.framework/libLAPACK.dylib";
#elif defined(_ENABLE_PYTHON)
    std::string get_scipylibs_path_() {
        // Exclusively for python calls
        // LCOV_EXCL_START
        std::string currentPathStr(getPath());
        std::string site_packages_str("site-packages/");
        std::string scipyPathStr;

        std::size_t str_pos = currentPathStr.find(site_packages_str);
        if (str_pos != std::string::npos) {
            scipyPathStr =
                currentPathStr.substr(0, str_pos + site_packages_str.size());
            scipyPathStr += "scipy.libs";
        }

        if (std::filesystem::exists(scipyPathStr)) {
            try {
                // convert the relative path to absolute path
                scipyPathStr =
                    std::filesystem::canonical(scipyPathStr).string();
            } catch (const std::exception &err) {
                std::cerr << "Canonical path for scipy.libs"
                          << " threw exception:\n"
                          << err.what() << '\n';
            }
        } else {
            try {
                scipyPathStr = currentPathStr + "/../../scipy.libs/";
                // convert the relative path to absolute path
                scipyPathStr =
                    std::filesystem::canonical(scipyPathStr).string();
            } catch (const std::exception &err) {
                std::cerr << "Canonical path for scipy.libs"
                          << " threw exception:\n"
                          << err.what() << '\n';
            }
        }
        return scipyPathStr;
        // LCOV_EXCL_STOP
    }
#elif not defined(_ENABLE_PYTHON)
    std::string get_scipylibs_path_worker_() {
        pybind11::object scipy_module =
            pybind11::module::import("scipy").attr("__file__");

        std::string scipy_path = pybind11::str(scipy_module);
        std::string scipy_lib_path =
            scipy_path.substr(0, scipy_path.find("scipy")) + "scipy.libs";

        return scipy_lib_path;
    }

    /**
     * @brief Get the path to the scipy.libs package.
     *
     * This function will return the path to the scipy.libs package. It will
     * first try to get the path from the current Python environment. If the
     * Python environment is not initialized, it will initialize it and get the
     * path.
     *
     * @return std::string The path to the scipy.libs package.
     */
    std::string get_scipylibs_path_() {
        pybind11::scoped_interpreter scope_guard{};
        return get_scipylibs_path_worker_();
    }
#endif
    /**
     * @brief BLASLibLoaderManager.
     *
     * This function will initialize the BLASLibLoaderManager by searching for
     * the BLAS libraries in the given path.
     *
     * @param blas_lib_path_str The path to the BLAS libraries.
     */
    void init_helper_(const std::string &blas_lib_path_str) {
        std::filesystem::path scipyLibsPath(blas_lib_path_str);

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

        scipy_prefix_ = std::find_if(availableLibs.begin(), availableLibs.end(),
                                     [](const auto &lib) {
                                         return lib.find("scipy_openblas") !=
                                                std::string::npos;
                                     }) != availableLibs.end();

        blasLib = blasLibs.back();
    }
    /**
     * @brief BLASLibLoaderManager.
     *
     * This function will initialize the BLASLibLoaderManager by searching for
     * the BLAS libraries in the given path.
     *
     * @param blaslib_path The path to the BLAS libraries.
     */
    explicit BLASLibLoaderManager(
        [[maybe_unused]] const std::string &blaslib_path_str) {
#if defined(__APPLE__)
        // On macOS, use the default BLAS library path.
        blasLib = std::make_shared<SharedLibLoader>(scipy_lib_path_macos_str);
#elif defined(_ENABLE_PYTHON)
        // Given that the BLAS library path is provided by the Python layer, use
        // the provided path.
        std::filesystem::path blaslib_path(blaslib_path_str);
        PL_ABORT_IF_NOT(std::filesystem::exists(blaslib_path),
                        "The provided BLAS library path does not exist.");
        init_helper_(blaslib_path_str);
#else
        // Given that the BLAS library path is not provided by the C++
        // layer, use the path found by pybind11.
        const std::string scipyPathStr = get_scipylibs_path_();
        init_helper_(scipyPathStr);
#endif
    }

  public:
    BLASLibLoaderManager(BLASLibLoaderManager &&) = delete;
    BLASLibLoaderManager(const BLASLibLoaderManager &) = delete;
    BLASLibLoaderManager &operator=(const BLASLibLoaderManager &) = delete;
    BLASLibLoaderManager operator=(const BLASLibLoaderManager &&) = delete;

    static BLASLibLoaderManager &
    getInstance(const std::string &blaslib_path = {}) {
        static BLASLibLoaderManager instance(blaslib_path);
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
    auto getBLASLib() -> SharedLibLoader * { return blasLib.get(); }

    /**
     * @brief Get the BLAS libraries.
     *
     * This function will return the BLAS libraries.
     *
     * @return std::vector<SharedLibLoader*> The BLAS libraries.
     */
    [[nodiscard]] auto getScipyPrefix() const -> bool { return scipy_prefix_; }
};
} // namespace Pennylane::Util