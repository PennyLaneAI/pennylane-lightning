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
#include "ConstantUtil.hpp"
#include "Macros.hpp"
#include "RuntimeInfo.hpp"

#include <benchmark/benchmark.h>

#include <string>

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
template <typename T> struct PrecisionToStr;

template <> struct PrecisionToStr<float> {
    constexpr static std::string_view value = "float";
};

template <> struct PrecisionToStr<double> {
    constexpr static std::string_view value = "double";
};

template <typename T>
constexpr static auto precision_to_str = PrecisionToStr<T>::value;

/**
 * @brief Generate neighboring wires from a start index
 *
 * @param start_idx Start index.
 * @param num_qubits Number of qubits.
 * @param num_wires Number of wires to be considered.
 *
 * @return std::vector<size_t>
 */
inline auto generateNeighboringWires(size_t start_idx, size_t num_qubits,
                                     size_t num_wires) -> std::vector<size_t> {
    std::vector<size_t> v;
    v.reserve(num_wires);
    for (size_t k = 0; k < num_wires; k++) {
        v.emplace_back((start_idx + k) % num_qubits);
    }
    return v;
}

/**
 * @brief Generate distinct wires.
 *
 * @tparam RandomEngine Random number generator engine.
 * @param num_qubits Number of qubits.
 * @param num_wires Number of wires.
 *
 * @return std::vector<size_t>
 */
template <typename RandomEngine>
inline auto generateDistinctWires(RandomEngine &eng, size_t num_qubits,
                                  size_t num_wires) -> std::vector<size_t> {
    std::vector<size_t> v(num_qubits, 0);
    std::iota(v.begin(), v.end(), 0);
    shuffle(v.begin(), v.end(), eng);
    return {v.begin(), v.begin() + num_wires};
}

inline auto addCompileInfo() {
    using namespace Pennylane;
    constexpr auto compiler_name =
        Util::lookup(Util::Constant::compiler_names, Util::Constant::compiler);
    benchmark::AddCustomContext("Compiler::Name", std::string(compiler_name));
    benchmark::AddCustomContext(
        "Compiler::Version",
        std::string(
            Util::Constant::getCompilerVersion<Util::Constant::compiler>()));
}

constexpr auto boolToStr(bool val) -> std::string_view {
    return val ? "True" : "False";
}

inline auto addRuntimeInfo() {
    using Pennylane::Util::RuntimeInfo;
    benchmark::AddCustomContext("CPU::Vendor", RuntimeInfo::vendor());
    benchmark::AddCustomContext("CPU::Brand", RuntimeInfo::brand());
    benchmark::AddCustomContext("CPU::AVX",
                                std::string{boolToStr(RuntimeInfo::AVX())});
    benchmark::AddCustomContext("CPU::AVX2",
                                std::string{boolToStr(RuntimeInfo::AVX2())});
    benchmark::AddCustomContext("CPU::AVX512F",
                                std::string{boolToStr(RuntimeInfo::AVX512F())});
}
