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
#include <string>

#if defined(__APPLE__) || defined(__linux__)
#include <dlfcn.h>
#elif defined(_MSC_VER)
#include <windows.h>
#endif

#include "Error.hpp"

namespace Pennylane::Util {
/**
 * Dynamic shared library loading wrapper class
 */
// Adapted from Catalyst
// (https://github.com/PennyLaneAI/catalyst/blob/f016a31f69d1b8a84bc9612af1bc64f0575506e9/runtime/lib/capi/ExecutionContext.hpp#L75)
class SharedLibLoader final {
  private:
#if defined(__APPLE__) || defined(__linux__)
    void *handle_{nullptr};
#elif defined(_MSC_VER)
    HMODULE handle_{nullptr};
#endif

  public:
    SharedLibLoader();
    explicit SharedLibLoader(const std::string &filename) {
#if defined(__APPLE__)
        auto rtld_flags = RTLD_LAZY;
#elif defined(__linux__)
        auto rtld_flags = RTLD_LAZY | RTLD_NODELETE;
#endif

#if defined(__APPLE__) || defined(__linux__)
        handle_ = dlopen(filename.c_str(), rtld_flags);
        // This allows users to use pre-installed LAPACK package
        PL_ABORT_IF(!handle_, dlerror());
#elif defined(_MSC_VER)
#pragma warning(push, 0)
        handle_ = LoadLibrary(filename.c_str());
#pragma warning(pop)
        PL_ABORT_IF(!handle_, std::to_string(GetLastError()));
#endif
    }

    ~SharedLibLoader() {
#if defined(__APPLE__) || defined(__linux__)
        dlclose(handle_);
#elif defined(_MSC_VER)
        FreeLibrary(handle_);
#endif
    }

    void *getHandle() { return handle_; }

    void *getSymbol(const std::string &symbol) {
#if defined(__APPLE__) || defined(__linux__)
        void *sym = dlsym(handle_, symbol.c_str());
        PL_ABORT_IF(!sym, dlerror());
#elif defined(_MSC_VER)
        void *sym = GetProcAddress(handle_, symbol.c_str());
        PL_ABORT_IF(!handle_, std::to_string(GetLastError()));
#endif
        return sym;
    }
};

} // namespace Pennylane::Util