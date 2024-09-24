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
 * Dynamic shared library functions API wrapper.
 */
#pragma once
#include <mutex>
#include <string>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#define HANDLE_TYPE void *
#define PL_DLOPEN(NAME, ARG) dlopen(NAME, ARG)
#define PL_DLERROR() dlerror()
#define PL_DLCLOSE(NAME) dlclose(NAME)
#define PL_DLSYS(NAME, SYMBOL) dlsym(NAME, SYMBOL)

#else
#define NOMINMAX
#include <windows.h>
#define HANDLE_TYPE HMODULE
#define PL_DLOPEN(NAME, ARG) LoadLibrary(NAME)
#define PL_DLERROR() GetLastError()
#define PL_DLCLOSE(NAME) FreeLibrary(NAME)
#define PL_DLSYS(NAME, SYMBOL) GetProcAddress(NAME, SYMBOL)
#endif

#include "Error.hpp"

namespace Pennylane::Util {
/**
 * Dynamic shared library loading wrapper class
 *
 * This class is adapted from Catalyst
 * https://github.com/PennyLaneAI/catalyst/blob/f016a31f69d1b8a84bc9612af1bc64f0575506e9/runtime/lib/capi/ExecutionContext.hpp#L75
 *
 */

// Ignore invalid warnings for compile-time checks
// NOLINTBEGIN
class SharedLibLoader final {
  private:
    HANDLE_TYPE handle_;
    std::mutex mtx_;

  public:
    SharedLibLoader();
    explicit SharedLibLoader(const std::string &filename) {
        const std::lock_guard<std::mutex> lock(mtx_);
        handle_ = PL_DLOPEN(filename.c_str(), RTLD_LAZY);
        PL_ABORT_IF(!handle_, PL_DLERROR());
    }

    ~SharedLibLoader() noexcept {
        const std::lock_guard<std::mutex> lock(mtx_);
        PL_DLCLOSE(handle_);
    }

    HANDLE_TYPE getHandle() { return handle_; }

    template <typename FunPtr> FunPtr getSymbol(const std::string &symbol) {
        return reinterpret_cast<FunPtr>(PL_DLSYS(handle_, symbol.c_str()));
    }
};
// NOLINTEND

} // namespace Pennylane::Util
