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

using namespace Pennylane;

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
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(create_random_cmplx_vector<double>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

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
            Util::ConstSum<T>,
            static_cast<std::complex<T> (*)(std::complex<T>, std::complex<T>)>(
                &Util::ConstMult<T>));
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(std_innerProd_cmplx<float>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(std_innerProd_cmplx<double>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

/**
 * @brief Benchmark Util::omp_innerProd for two vectors of complex
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
        // Create indirection to avoid GCC issue with AVX512 compilation
        std::complex<T> *res_ptr = &res;

        Util::omp_innerProd(vec1.data(), vec2.data(), *res_ptr, sz);
        benchmark::DoNotOptimize(res_ptr);
    }
}
BENCHMARK(omp_innerProd_cmplx<float>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(omp_innerProd_cmplx<double>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

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
        // Create indirection to avoid GCC issue with AVX512 compilation
        std::complex<T> *res_ptr = &res;

        if constexpr (std::is_same_v<T, float>) {
            cblas_cdotc_sub(sz, vec1.data(), 1, vec2.data(), 1, res_ptr);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_zdotc_sub(sz, vec1.data(), 1, vec2.data(), 1, res_ptr);
        }

        benchmark::DoNotOptimize(res_ptr);
    }
}
BENCHMARK(blas_innerProd_cmplx<float>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(blas_innerProd_cmplx<double>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);
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
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(naive_transpose_cmplx<double>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

/**
 * @brief Benchmark Util::CFTranspose for a randomly generated matrix
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

        Util::CFTranspose<T, BLOCKSIZE>(mat1.data(), mat2.data(), sz, sz, 0, sz,
                                        0, sz);
        benchmark::DoNotOptimize(mat2[sz * sz - 1]);
    }
}
BENCHMARK(cf_transpose_cmplx<float, 16>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(cf_transpose_cmplx<double, 16>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(cf_transpose_cmplx<float, 32>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

BENCHMARK(cf_transpose_cmplx<double, 32>)
    ->RangeMultiplier(1U << 3)
    ->Range(1U << 5, 1U << 10);

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

        Util::omp_matrixVecProd(mat.data(), vec1.data(), vec2.data(), sz, sz,
                                Trans::NoTranspose);
        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(omp_matrixVecProd_cmplx<float>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

BENCHMARK(omp_matrixVecProd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

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
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

BENCHMARK(blas_matrixVecProd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);
#endif

//***********************************************************************//
//                         Matrix-Matrix Product
//***********************************************************************//

/**
 * @brief Benchmark Util::omp_matrixMatProd for two randomly
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

    const auto m_right_tr = Util::Transpose(m_right, sz, sz);

    for (auto _ : state) {
        std::vector<std::complex<T>> m_out(sz * sz);

        Util::omp_matrixMatProd(m_left.data(), m_right_tr.data(), m_out.data(),
                                sz, sz, sz, Trans::Transpose);
        benchmark::DoNotOptimize(m_out[sz * sz - 1]);
    }
}
BENCHMARK(omp_matrixMatProd_cmplx<float>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

BENCHMARK(omp_matrixMatProd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

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
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);

BENCHMARK(blas_matrixMatProd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4, 1U << 8);
#endif

//***********************************************************************//
//                         Scale and add
//***********************************************************************//

/**
 * @brief Benchmark scaleAndAdd function implemented in the standard way
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void std_scaleAndAdd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> vec1;
    std::vector<std::complex<T>> vec2;
    std::complex<T> scale{std::cos(static_cast<T>(0.4123)),
                          std::sin(static_cast<T>(0.4123))};

    for (size_t i = 0; i < sz; i++) {
        vec1.push_back({distr(eng), distr(eng)});
    }
    for (size_t i = 0; i < sz; i++) {
        vec2.push_back({distr(eng), distr(eng)});
    }

    for (auto _ : state) {
        for (size_t i = 0; i < sz; i++) {
            vec2[i] += scale * vec1[i];
        }
        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(std_scaleAndAdd_cmplx<float>)
    ->RangeMultiplier(1U << 2U)
    ->Range(1U << 4U, 1U << 20U);

BENCHMARK(std_scaleAndAdd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4U, 1U << 20U);

/**
 * @brief Benchmark PennyLane::Util::omp_scaleAndAdd for a randomly generated
 * matrix and vector of complex numbers.
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void omp_scaleAndAdd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> vec1;
    std::vector<std::complex<T>> vec2;
    std::complex<T> scale{std::cos(static_cast<T>(0.4123)),
                          std::sin(static_cast<T>(0.4123))};

    for (size_t i = 0; i < sz; i++) {
        vec1.push_back({distr(eng), distr(eng)});
    }
    for (size_t i = 0; i < sz; i++) {
        vec2.push_back({distr(eng), distr(eng)});
    }

    for (auto _ : state) {
        Util::omp_scaleAndAdd(sz, scale, vec1.data(), vec2.data());
        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(omp_scaleAndAdd_cmplx<float>)
    ->RangeMultiplier(1U << 2U)
    ->Range(1U << 4U, 1U << 20U);

BENCHMARK(omp_scaleAndAdd_cmplx<double>)
    ->RangeMultiplier(1U << 2U)
    ->Range(1U << 4U, 1U << 20U);

#if __has_include(<cblas.h>) && defined _ENABLE_BLAS
/**
 * @brief Benchmark blas_scaleAndAdd
 *
 * @tparam T Floating point precision type.
 */
template <class T> static void blas_scaleAndAdd_cmplx(benchmark::State &state) {
    std::random_device rd;
    std::mt19937_64 eng(rd());
    std::uniform_real_distribution<T> distr;
    const auto sz = static_cast<size_t>(state.range(0));

    std::vector<std::complex<T>> vec1;
    std::vector<std::complex<T>> vec2;
    std::complex<T> scale{std::cos(static_cast<T>(0.4123)),
                          std::sin(static_cast<T>(0.4123))};

    for (size_t i = 0; i < sz; i++) {
        vec1.push_back({distr(eng), distr(eng)});
    }
    for (size_t i = 0; i < sz; i++) {
        vec2.push_back({distr(eng), distr(eng)});
    }

    for (auto _ : state) {
        Util::blas_scaleAndAdd(sz, scale, vec1.data(), vec2.data());
        benchmark::DoNotOptimize(vec2[sz - 1]);
    }
}
BENCHMARK(blas_scaleAndAdd_cmplx<float>)
    ->RangeMultiplier(1U << 2U)
    ->Range(1U << 4U, 1U << 20U);

BENCHMARK(blas_scaleAndAdd_cmplx<double>)
    ->RangeMultiplier(1U << 2)
    ->Range(1U << 4U, 1U << 20U);
#endif
