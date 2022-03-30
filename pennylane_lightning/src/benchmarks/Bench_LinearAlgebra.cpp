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
#include <limits>
#include <random>

#include <LinearAlgebra.hpp>

#include <benchmark/benchmark.h>

/**
 * @brief Benchmark generating a vector of random complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void create_random_cmplx_vector(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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

//***********************************************************************//
//                            Inner Product
//***********************************************************************//

/**
 * @brief Benchmark std::inner_product for two vectors of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void std_innerProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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

/**
 * @brief Benchmark Pennylane::Util::omp_innerProd for two vectors of complex
 * numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void omp_innerProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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
/**
 * @brief Benchmark cblas_cdotc_sub and cblas_zdotc_sub for two vectors of
 * complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void blas_innerProd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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

//***********************************************************************//
//                           Matrix Transpose
//***********************************************************************//

/**
 * @brief Benchmark naive matrix transpose for a randomly generated matrix
 * of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void naive_transpose_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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

/**
 * @brief Benchmark Pennylane::Util::CFTranspose for a randomly generated matrix
 * of complex numbers.
 *
 * @tparam T Floating point precision type.
 * @tparam BLOCKSIZE Size of submatrices in the blocking technique.
 */
template <class T, size_t BLOCKSIZE>
static void cf_transpose_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> mat1;
    for (size_t i = 0; i < sz * sz; i++)
        mat1.push_back({distr(eng), distr(eng)});

    for (auto _ : state) {
        std::vector<std::complex<T>> mat2(sz * sz);

        Pennylane::Util::CFTranspose<T, BLOCKSIZE>(mat1.data(), mat2.data(), sz,
                                                   sz, 0, sz, 0, sz);
        benchmark::DoNotOptimize(mat2[sz * sz - 1]);
    }
}
BENCHMARK(cf_transpose_cmplx<float, 16>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(cf_transpose_cmplx<double, 16>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(cf_transpose_cmplx<float, 32>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

BENCHMARK(cf_transpose_cmplx<double, 32>)
    ->RangeMultiplier(1l << 3)
    ->Range(1l << 5, 1l << 10);

//***********************************************************************//
//                         Matrix-Vector Product
//***********************************************************************//

/**
 * @brief Benchmark PennyLane::Util::omp_matrixVecProd for a randomly generated
 * matrix and vector of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void omp_matrixVecProd_cmplx(benchmark::State &state) {
    using Pennylane::Util::Trans;
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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
/**
 * @brief Benchmark cblas_cgemv and cblas_zgemv for a randomly generated
 * matrix and vector of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void blas_matrixVecProd_cmplx(benchmark::State &state) {
    using Pennylane::Util::Trans;
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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

//***********************************************************************//
//                         Matrix-Matrix Product
//***********************************************************************//

/**
 * @brief Benchmark Pennylane::Util::omp_matrixMatProd for two randomly
 * generated matrices of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void omp_matrixMatProd_cmplx(benchmark::State &state) {
    using Pennylane::Util::Trans;
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> m_left;
    for (size_t i = 0; i < sz * sz; i++)
        m_left.push_back({distr(eng), distr(eng)});

    std::vector<std::complex<T>> m_right;
    for (size_t i = 0; i < sz * sz; i++)
        m_right.push_back({distr(eng), distr(eng)});

    const auto m_right_tr = Pennylane::Util::Transpose(m_right, sz, sz);

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
/**
 * @brief Benchmark cblas_cgemm and cblas_zgemm for two randomly
 * generated matrices of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T>
static void blas_matrixMatProd_cmplx(benchmark::State &state) {
    using Pennylane::Util::Trans;
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

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
