#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>

#include "StateVectorManaged.hpp"

#include <benchmark/benchmark.h>

template <class T>
static void create_random_cmplx_vector(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;

    auto sz = static_cast<size_t>(state.range(0));
    for (auto _ : state) {
        std::vector<std::complex<T>> vec;
        for (size_t i = 0; i < sz; i++) {
            vec.push_back({distr(eng), distr(eng)});
        }
        benchmark::DoNotOptimize(vec.size());
    }
}
BENCHMARK(create_random_cmplx_vector<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(create_random_cmplx_vector<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);
