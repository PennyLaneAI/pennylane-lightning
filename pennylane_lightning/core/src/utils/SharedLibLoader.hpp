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
#include <string>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#define HANDLE_TYPE void *
#define PL_DLOPEN(NAME, ARG) dlopen(NAME, ARG)
#define PL_DLERROR() dlerror()
#define PL_DLCLOSE(NAME) dlclose(NAME)
#define PL_DLSYS(NAME, SYMBOL) dlsym(NAME, SYMBOL)
#elif defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#define HANDLE_TYPE HMODULE
#define PL_DLOPEN(NAME, ARG) LoadLibrary(NAME)
#define PL_DLERROR() std::to_string(GetLastError())
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
    HANDLE_TYPE handle_{nullptr};

  public:
    SharedLibLoader();
    explicit SharedLibLoader(const std::string &filename) {
        // NOTE: RTLD_NODELETE flag is a temporary solution. It could be
        // problematic if the shared library is not static stored in memory and
        // runtime unloaded is needed. Come back to this later.
        handle_ = PL_DLOPEN(filename.c_str(), RTLD_LAZY | RTLD_NODELETE);
        PL_ABORT_IF(!handle_, PL_DLERROR());
    }

    ~SharedLibLoader() noexcept { PL_DLCLOSE(handle_); }

    HANDLE_TYPE getHandle() { return handle_; }

    template <typename FunPtr> FunPtr getSymbol(const std::string &symbol) {
        FunPtr func_ptr =
            reinterpret_cast<FunPtr>(PL_DLSYS(handle_, symbol.c_str()));
        PL_ABORT_IF(!func_ptr, PL_DLERROR());
        return func_ptr;
    }
};
// NOLINTEND

} // namespace Pennylane::Util
