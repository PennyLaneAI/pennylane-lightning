#include <limits>
#include <random>

#include <BitUtil.hpp>

#include <benchmark/benchmark.h>

static void generate_uniform_random_number(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long long> distr;
    for (auto _ : state) {
        benchmark::DoNotOptimize(distr(eng));
    }
}
BENCHMARK(generate_uniform_random_number);

static void naive_popcount(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(Pennylane::Util::Internal::countBit1(val));
    }
}
BENCHMARK(naive_popcount);

#if defined(__GNUC__) || defined(__clang__)
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
#endif

#if defined(_MSC_VER)
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

static void naive_log2PerfectPower(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<unsigned long> distr;
    for (auto _ : state) {
        auto val = static_cast<unsigned long>(distr(eng));
        benchmark::DoNotOptimize(
            Pennylane::Util::Internal::countTrailing0(val));
    }
}
BENCHMARK(naive_log2PerfectPower);

#if defined(__GNUC__) || defined(__clang__)
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

static void lightning_isPerfectPowerOf2(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_int_distribution<size_t> distr;
    for (auto _ : state) {
        auto val = static_cast<size_t>(distr(eng));
        benchmark::DoNotOptimize(Pennylane::Util::isPerfectPowerOf2(val));
    }
}
BENCHMARK(lightning_isPerfectPowerOf2);

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
