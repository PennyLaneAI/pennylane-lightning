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

/**
 * @brief Create an ordered list of {lo, lo + sum, ..., lo + k * sum, hi}
 * to be used by benchmark::ArgsProduct.
 *
 * @param lo int64_t
 * @param hi int64_t
 * @param sum int64_t
 *
 * @return std::vector<int64_t>
 */
std::vector<int64_t> CreateDenseRange(int64_t lo, int64_t hi, int sum) {
    std::vector<int64_t> v;
    int64_t i = lo;
    for (; i < hi; i += sum) {
        v.push_back(i);
    }
    v.push_back(hi);
    return v;
}
