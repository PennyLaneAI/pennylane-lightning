// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <benchmark/benchmark.h>

/**
 * @brief A benchmark macro to register func<t>(...)
 * using benchmark::internal::FunctionBenchmark.
 */
#define BENCHMARK_APPLYOPS(func, t, test_case_name, ...)                       \
    BENCHMARK_PRIVATE_DECLARE(func) =                                          \
        (benchmark::internal::RegisterBenchmarkInternal(                       \
            new benchmark::internal::FunctionBenchmark(                        \
                #func "<" #t ">"                                               \
                      "/" #test_case_name,                                     \
                [](benchmark::State &st) { func<t>(st, __VA_ARGS__); })))
