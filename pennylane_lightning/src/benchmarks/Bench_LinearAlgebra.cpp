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
    ->Range(1l << 5, 1l << 10);

BENCHMARK(create_random_cmplx_vector<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

template <class T> static void std_innerProd_cmplx(benchmark::State &state) {
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
BENCHMARK(std_innerProd_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(std_innerProd_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

template <class T> static void omp_innerProd_cmplx(benchmark::State &state) {
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
        std::complex<T> res(.0, .0);

        Pennylane::Util::omp_innerProd(vec1.data(), vec2.data(), res, sz);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(omp_innerProd_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(omp_innerProd_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
template <class T> static void blas_innerProd_cmplx(benchmark::State &state) {
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
        std::complex<T> res(.0, .0);

        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotc_sub(sz, vec1.data(), 1, vec2.data(), 1, &res);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotc_sub(sz, vec1.data(), 1, vec2.data(), 1, &res);
        }

        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(blas_innerProd_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(blas_innerProd_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);
#endif

template <class T> static void naive_transpose_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat1;
    for (size_t i = 0; i < sz * sz; i++)
        mat1.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::vector<std::complex<T>> mat2(sz * sz);

        for (size_t r = 0; r < sz; r++) {
            for (size_t s = 0; s < sz; s++) {
                mat2[s * sz + r] = mat1[r * sz + s];
            }
        }

        benchmark::DoNotOptimize(mat2[sz * sz - 1]);
    }
}
BENCHMARK(naive_transpose_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(naive_transpose_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

template <class T> static void cf_transpose_b16_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat1;
    for (size_t i = 0; i < sz * sz; i++)
        mat1.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::vector<std::complex<T>> mat2(sz * sz);

        Pennylane::Util::CFTranspose<T, 16UL>(mat1.data(), mat2.data(), sz, sz,
                                              0, sz, 0, sz);
        benchmark::DoNotOptimize(mat2[sz * sz - 1]);
    }
}
BENCHMARK(cf_transpose_b16_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(cf_transpose_b16_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

template <class T> static void cf_transpose_b32_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat1;
    for (size_t i = 0; i < sz * sz; i++)
        mat1.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::vector<std::complex<T>> mat2(sz * sz);

        Pennylane::Util::CFTranspose<T, 32UL>(mat1.data(), mat2.data(), sz, sz,
                                              0, sz, 0, sz);
        benchmark::DoNotOptimize(mat2[sz * sz - 1]);
    }
}
BENCHMARK(cf_transpose_b32_cmplx<float>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(cf_transpose_b32_cmplx<double>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

template <class T>
static void omp_matrixVecProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat;
    for (size_t i = 0; i < sz * sz; i++)
        mat.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> vec1;
    for (size_t i = 0; i < sz; i++)
        vec1.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::vector<std::complex<T>> vec2(sz);

        Pennylane::Util::omp_matrixVecProd(mat.data(), vec1.data(), vec2.data(),
                                           sz, sz, Trans::NoTranspose);
        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(omp_matrixVecProd_cmplx<float>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

BENCHMARK(omp_matrixVecProd_cmplx<double>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
template <class T>
static void blas_matrixVecProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat;
    for (size_t i = 0; i < sz * sz; i++)
        mat.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> vec1;
    for (size_t i = 0; i < sz; i++)
        vec1.push_back({distr(eng), distr(eng)});

    const auto tr = static_cast<CBLAS_TRANSPOSE>(Trans::NoTranspose);
    constexpr std::complex<T> co{1, 0};
    constexpr std::complex<T> cz{0, 0};

    for (auto _ : state) {
        std::vector<std::complex<T>> vec2(sz);

        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemv(CblasRowMajor, tr, sz, sz, &co, mat.data(), sz,
                        vec1.data(), 1, &cz, vec2.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemv(CblasRowMajor, tr, sz, sz, &co, mat.data(), sz,
                        vec1.data(), 1, &cz, vec2.data(), 1);
        }

        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(blas_matrixVecProd_cmplx<float>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

BENCHMARK(blas_matrixVecProd_cmplx<double>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);
#endif

template <class T>
static void omp_matrixMatProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> m_left;
    for (size_t i = 0; i < sz * sz; i++)
        m_left.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> m_right;
    for (size_t i = 0; i < sz * sz; i++)
        m_right.push_back({distr(eng), distr(eng)});

    auto m_right_tr = Pennylane::Util::Transpose(m_right, sz, sz);

    for (auto _ : state) {
        std::vector<std::complex<T>> m_out(sz * sz);

        Pennylane::Util::omp_matrixMatProd(m_left.data(), m_right_tr.data(),
                                           m_out.data(), sz, sz, sz,
                                           Trans::Transpose);
        benchmark::DoNotOptimize(m_out[sz * sz - 1]);
    }
}
BENCHMARK(omp_matrixMatProd_cmplx<float>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

BENCHMARK(omp_matrixMatProd_cmplx<double>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
template <class T>
static void blas_matrixMatProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> m_left;
    for (size_t i = 0; i < sz * sz; i++)
        m_left.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> m_right;
    for (size_t i = 0; i < sz * sz; i++)
        m_right.push_back({distr(eng), distr(eng)});

    const auto tr = static_cast<CBLAS_TRANSPOSE>(Trans::NoTranspose);
    constexpr std::complex<T> co{1, 0};
    constexpr std::complex<T> cz{0, 0};

    for (auto _ : state) {
        std::vector<std::complex<T>> m_out(sz * sz);

        if constexpr (std::is_same_v<T, float>) {
            cblas_cgemm(CblasRowMajor, CblasNoTrans, tr, sz, sz, sz, &co,
                        m_left.data(), sz, m_right.data(), sz, &cz,
                        m_out.data(), sz);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, tr, sz, sz, sz, &co,
                        m_left.data(), sz, m_right.data(), sz, &cz,
                        m_out.data(), sz);
        }
        benchmark::DoNotOptimize(m_out[sz * sz - 1]);
    }
}
BENCHMARK(blas_matrixMatProd_cmplx<float>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);

BENCHMARK(blas_matrixMatProd_cmplx<double>)
    ->RangeMultiplier(1l << 2)
    ->Range(1l << 4, 1l << 8);
#endif
