#include <limits>
#include <random>

#include <LinearAlgebra.hpp>

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
    ->Range(1l << 10, 1l << 15);

BENCHMARK(create_random_cmplx_vector<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 10, 1l << 15);

template <class T> static void std_inner_product(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> vec1;
    for (size_t i = 0; i < sz; i++)
        vec1.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> vec2;
    for (size_t i = 0; i < sz; i++)
        vec2.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::complex<T> res = std::inner_product(
            vec1.data(), vec1.data() + sz, vec2.data(), std::complex<T>(),
            Pennylane::Util::ConstSum<T>,
            static_cast<std::complex<T> (*)(std::complex<T>, std::complex<T>)>(
                &Pennylane::Util::ConstMult<T>));
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(std_inner_product<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 10, 1l << 15);

BENCHMARK(std_inner_product<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 10, 1l << 15);
