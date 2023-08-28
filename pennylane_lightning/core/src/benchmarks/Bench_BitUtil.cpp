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
#include <bit>
#include <limits>
#include <random>

#include <BitUtil.hpp>

#include <benchmark/benchmark.h>

/**
 * @brief Benchmark generating an uniform random integer.
 */
static void generate_uniform_random_number(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long long> distr;
    for (auto _ : state) {
        benchmark::DoNotOptimize(distr(eng));
    }
}
BENCHMARK(generate_uniform_random_number);

/**
 * @brief Benchmark std popcount.
 */
static void std_popcount(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(std::popcount(val));
    }
}
BENCHMARK(std_popcount);

#if defined(__GNUC__) || defined(__clang__)
/**
 * @brief Benchmark builtin popcount.
 */
static void builtin_popcount(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(__builtin_popcountl(val));
    }
}
BENCHMARK(builtin_popcount);
;
#endif

#if defined(_MSC_VER)
/**
 * @brief Benchmark builtin popcount.
 */
static void msv_builtin_popcount(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<uint64_t> distr;
    for (auto _ : state) {
        auto val = static_cast<uint64_t>(distr(eng));
        benchmark::DoNotOptimize(__popcnt64(val));
    }
}
BENCHMARK(msv_builtin_popcount);
#endif

/**
 * @brief Benchmark std has_single_bit
 */
static void std_log2PerfectPower(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(std::has_single_bit(val));
    }
}
BENCHMARK(std_log2PerfectPower);

#if defined(__GNUC__) || defined(__clang__)
/**
 * @brief Benchmark builtin log2PerfectPower using __builtin_ctzl.
 */
static void builtin_log2PerfectPower(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(__builtin_ctzl(val));
    }
}
BENCHMARK(builtin_log2PerfectPower);
#endif

#if defined(_MSC_VER)
/**
 * @brief Benchmark builtin log2PerfectPower using __lzcnt64.
 */
static void msv_builtin_log2PerfectPower(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<uint64_t> distr;
    for (auto _ : state) {
        auto val = static_cast<uint64_t>(distr(eng));
        benchmark::DoNotOptimize(63 - __lzcnt64(val));
    }
}
BENCHMARK(msv_builtin_log2PerfectPower);
#endif

/**
 * @brief Benchmark fillTrailingOnes in PennyLane-Lightning.
 */
static void lightning_fillTrailingOnes(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<size_t> distr;
    for (auto _ : state) {
        auto val = static_cast<size_t>(distr(eng));
        benchmark::DoNotOptimize(Pennylane::Util::fillTrailingOnes(val));
    }
}
BENCHMARK(lightning_fillTrailingOnes);

/**
 * @brief Benchmark fillLeadingOnes in PennyLane-Lightning.
 */
static void lightning_fillLeadingOnes(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<size_t> distr;
    for (auto _ : state) {
        auto val = static_cast<size_t>(distr(eng));
        benchmark::DoNotOptimize(Pennylane::Util::fillLeadingOnes(val));
    }
}
BENCHMARK(lightning_fillLeadingOnes);
