// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file MPSLinAlg.hpp
 * Linear algebra used by the Kokkos MPS backend (lightning.tensornet):
 *   - a batched, strided complex GEMM written with Kokkos (runs on every
 *     Kokkos execution space);
 *   - a host SVD: LAPACK ?gesvd from scipy-openblas (loaded at run time
 *     through Lightning's BLASLibLoaderManager) when SCIPY_OPENBLAS_ENABLED
 *     is defined, and a one-sided Jacobi SVD otherwise (or if LAPACK fails).
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "Error.hpp"

#ifdef SCIPY_OPENBLAS_ENABLED
#include "BLASLibLoaderManager.hpp"
#endif

namespace Pennylane::LightningTensornet {

template <class fp_t> using KComplex = Kokkos::complex<fp_t>;
template <class fp_t> using KVector = Kokkos::View<Kokkos::complex<fp_t> *>;

/// Operation applied to a GEMM operand
enum class Trans : int {
    N = 0, ///< as is
    T = 1, ///< transpose
    C = 2, ///< conjugate transpose
};

/**
 * @brief Batched strided GEMM on row-major matrices:
 * C[b] = op(A[b]) * op(B[b]) for b in [0, batch).
 *
 * Element (i, k) of A[b] (before op) is A(offA + b * strideA + i * lda + k),
 * and similarly for B and C. A stride of 0 broadcasts an operand.
 *
 * @param m Rows of op(A) and C.
 * @param n Columns of op(B) and C.
 * @param k Columns of op(A) / rows of op(B).
 */
template <class fp_t>
void batchedGemm(Trans ta, Trans tb, std::size_t batch, std::size_t m,
                 std::size_t n, std::size_t k, const KVector<fp_t> &A,
                 std::size_t offA, std::size_t lda, std::size_t strideA,
                 const KVector<fp_t> &B, std::size_t offB, std::size_t ldb,
                 std::size_t strideB, const KVector<fp_t> &C, std::size_t offC,
                 std::size_t ldc, std::size_t strideC) {
    using CT = KComplex<fp_t>;
    if (batch == 0 || m == 0 || n == 0) {
        return;
    }
    const int oa = static_cast<int>(ta);
    const int ob = static_cast<int>(tb);
    // Flat index with the column j fastest: consecutive GPU threads write
    // consecutive elements of C (an MDRangePolicy maps its first index to the
    // fastest GPU thread dimension, which is the batch here and often 1).
    const std::size_t mn = m * n;
    Kokkos::parallel_for(
        "mps_batched_gemm", Kokkos::RangePolicy<>(0, batch * mn),
        KOKKOS_LAMBDA(const std::size_t t) {
            const std::size_t b = t / mn;
            const std::size_t i = (t % mn) / n;
            const std::size_t j = t % n;
            const std::size_t a0 = offA + b * strideA;
            const std::size_t b0 = offB + b * strideB;
            CT acc{0, 0};
            for (std::size_t kk = 0; kk < k; kk++) {
                CT x = (oa == 0) ? A(a0 + i * lda + kk) : A(a0 + kk * lda + i);
                if (oa == 2) {
                    x = Kokkos::conj(x);
                }
                CT y = (ob == 0) ? B(b0 + kk * ldb + j) : B(b0 + j * ldb + kk);
                if (ob == 2) {
                    y = Kokkos::conj(y);
                }
                acc += x * y;
            }
            C(offC + b * strideC + i * ldc + j) = acc;
        });
}

/// Dense product op(A) (m x k) * op(B) (k x n) of packed row-major matrices
template <class fp_t>
KVector<fp_t> matmul(Trans ta, Trans tb, std::size_t m, std::size_t n,
                     std::size_t k, const KVector<fp_t> &A,
                     const KVector<fp_t> &B) {
    KVector<fp_t> C(Kokkos::view_alloc(Kokkos::WithoutInitializing, "mps_mm"),
                    m * n);
    const std::size_t lda = (ta == Trans::N) ? k : m;
    const std::size_t ldb = (tb == Trans::N) ? n : k;
    batchedGemm<fp_t>(ta, tb, 1, m, n, k, A, 0, lda, 0, B, 0, ldb, 0, C, 0, n,
                      0);
    return C;
}

/// Copy a device vector to a host std::vector
template <class fp_t>
std::vector<std::complex<fp_t>> toHost(const KVector<fp_t> &v) {
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v);
    std::vector<std::complex<fp_t>> out(v.size());
    for (std::size_t i = 0; i < v.size(); i++) {
        out[i] = std::complex<fp_t>{h(i).real(), h(i).imag()};
    }
    return out;
}

/// Copy host data to a new device vector
template <class fp_t>
KVector<fp_t> toDevice(const std::complex<fp_t> *p, std::size_t n,
                       const std::string &label = "mps_t") {
    KVector<fp_t> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, label), n);
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < n; i++) {
        h(i) = KComplex<fp_t>{p[i].real(), p[i].imag()};
    }
    Kokkos::deep_copy(d, h);
    return d;
}

template <class fp_t>
KVector<fp_t> toDevice(const std::vector<std::complex<fp_t>> &v,
                       const std::string &label = "mps_t") {
    return toDevice<fp_t>(v.data(), v.size(), label);
}

/// Deep copy of a device vector
template <class fp_t> KVector<fp_t> clone(const KVector<fp_t> &v) {
    KVector<fp_t> out(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, v.label()), v.size());
    Kokkos::deep_copy(out, v);
    return out;
}

/// Thin SVD M = U diag(S) Vh of a row-major m x n matrix (r = min(m, n))
template <class fp_t> struct SVDResult {
    std::vector<std::complex<fp_t>> U{};  ///< m x r
    std::vector<fp_t> S{};                ///< r, descending
    std::vector<std::complex<fp_t>> Vh{}; ///< r x n
};

/**
 * @brief One-sided (Hestenes) Jacobi SVD of a row-major m x n matrix.
 */
template <class fp_t>
SVDResult<fp_t> jacobiSVD(const std::vector<std::complex<fp_t>> &M,
                          std::size_t m, std::size_t n) {
    using C = std::complex<fp_t>;
    PL_ABORT_IF_NOT(M.size() == m * n, "jacobiSVD: size mismatch");
    if (m < n) {
        // SVD of M^dagger (n x m): M^dagger = U' S V'^dagger
        // => M = V' S U'^dagger
        std::vector<C> Mh(n * m);
        for (std::size_t i = 0; i < m; i++) {
            for (std::size_t j = 0; j < n; j++) {
                Mh[j * m + i] = std::conj(M[i * n + j]);
            }
        }
        auto t = jacobiSVD<fp_t>(Mh, n, m);
        const std::size_t r = m;
        SVDResult<fp_t> out;
        out.S = t.S;
        out.U.assign(m * r, C{0, 0});
        out.Vh.assign(r * n, C{0, 0});
        // t.Vh is r x m: V' = t.Vh^dagger (m x r)
        for (std::size_t i = 0; i < m; i++) {
            for (std::size_t j = 0; j < r; j++) {
                out.U[i * r + j] = std::conj(t.Vh[j * m + i]);
            }
        }
        // U'^dagger: t.U is n x r
        for (std::size_t j = 0; j < r; j++) {
            for (std::size_t i = 0; i < n; i++) {
                out.Vh[j * n + i] = std::conj(t.U[i * r + j]);
            }
        }
        return out;
    }
    // m >= n: orthogonalise the columns of M. Store column-major.
    std::vector<C> G(m * n);
    for (std::size_t i = 0; i < m; i++) {
        for (std::size_t j = 0; j < n; j++) {
            G[j * m + i] = M[i * n + j];
        }
    }
    std::vector<C> V(n * n, C{0, 0}); // column-major
    for (std::size_t j = 0; j < n; j++) {
        V[j * n + j] = C{1, 0};
    }
    const fp_t eps = std::numeric_limits<fp_t>::epsilon();
    const fp_t tol = eps * static_cast<fp_t>(std::max<std::size_t>(m, 8));
    for (int sweep = 0; sweep < 80; sweep++) {
        bool rotated = false;
        for (std::size_t p = 0; p + 1 < n; p++) {
            for (std::size_t q = p + 1; q < n; q++) {
                C *gp = &G[p * m];
                C *gq = &G[q * m];
                fp_t alpha = 0;
                fp_t beta = 0;
                C g{0, 0};
                for (std::size_t i = 0; i < m; i++) {
                    alpha += std::norm(gp[i]);
                    beta += std::norm(gq[i]);
                    g += std::conj(gp[i]) * gq[i];
                }
                const fp_t ag = std::abs(g);
                if (ag == fp_t{0} || ag <= tol * std::sqrt(alpha * beta)) {
                    continue;
                }
                rotated = true;
                const C e = g / ag;
                const fp_t zeta = (beta - alpha) / (2 * ag);
                const fp_t t =
                    (zeta >= 0 ? fp_t{1} : fp_t{-1}) /
                    (std::abs(zeta) + std::sqrt(fp_t{1} + zeta * zeta));
                const fp_t c = fp_t{1} / std::sqrt(fp_t{1} + t * t);
                const fp_t s = c * t;
                const C ce = std::conj(e);
                for (std::size_t i = 0; i < m; i++) {
                    const C x = gp[i];
                    const C y = gq[i] * ce;
                    gp[i] = c * x - s * y;
                    gq[i] = s * x + c * y;
                }
                C *vp = &V[p * n];
                C *vq = &V[q * n];
                for (std::size_t i = 0; i < n; i++) {
                    const C x = vp[i];
                    const C y = vq[i] * ce;
                    vp[i] = c * x - s * y;
                    vq[i] = s * x + c * y;
                }
            }
        }
        if (!rotated) {
            break;
        }
    }
    std::vector<fp_t> sig(n);
    for (std::size_t j = 0; j < n; j++) {
        fp_t s2 = 0;
        for (std::size_t i = 0; i < m; i++) {
            s2 += std::norm(G[j * m + i]);
        }
        sig[j] = std::sqrt(s2);
    }
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(
        order.begin(), order.end(),
        [&](std::size_t a, std::size_t b) { return sig[a] > sig[b]; });
    SVDResult<fp_t> out;
    out.S.resize(n);
    out.U.assign(m * n, C{0, 0});
    out.Vh.assign(n * n, C{0, 0});
    for (std::size_t jj = 0; jj < n; jj++) {
        const std::size_t j = order[jj];
        out.S[jj] = sig[j];
        const fp_t inv = sig[j] > fp_t{0} ? fp_t{1} / sig[j] : fp_t{0};
        for (std::size_t i = 0; i < m; i++) {
            out.U[i * n + jj] = G[j * m + i] * inv;
        }
        for (std::size_t i = 0; i < n; i++) {
            out.Vh[jj * n + i] = std::conj(V[j * n + i]);
        }
    }
    return out;
}

#ifdef SCIPY_OPENBLAS_ENABLED
namespace detail {
using zgesvdPtr = void (*)(const char *, const char *, const int *, const int *,
                           std::complex<double> *, const int *, double *,
                           std::complex<double> *, const int *,
                           std::complex<double> *, const int *,
                           std::complex<double> *, const int *, double *,
                           int *);
using cgesvdPtr = void (*)(const char *, const char *, const int *, const int *,
                           std::complex<float> *, const int *, float *,
                           std::complex<float> *, const int *,
                           std::complex<float> *, const int *,
                           std::complex<float> *, const int *, float *, int *);

/**
 * @brief LAPACK ?gesvd on a row-major matrix. Returns false on failure.
 *
 * The row-major m x n buffer is the column-major n x m matrix M^T. With
 * M^T = U1 S V1^H (column-major), M = (V1^H)^T S U1^T, and the column-major
 * buffers of V1^H (r x m) and U1 (n x r) are exactly the row-major buffers
 * of the left (m x r) and right (r x n) factors of M.
 */
template <class fp_t>
bool lapackSVD(std::vector<std::complex<fp_t>> A, std::size_t m, std::size_t n,
               SVDResult<fp_t> &out) {
    using C = std::complex<fp_t>;
    const int mm = static_cast<int>(n); // rows of M^T
    const int nn = static_cast<int>(m); // cols of M^T
    const int r = std::min(mm, nn);
    const int lda = mm;
    const int ldu = mm;
    const int ldvt = r;
    std::vector<fp_t> S(static_cast<std::size_t>(r));
    std::vector<C> Ubuf(static_cast<std::size_t>(ldu) * r);
    std::vector<C> VTbuf(static_cast<std::size_t>(ldvt) * nn);
    std::vector<fp_t> rwork(static_cast<std::size_t>(5 * r + 1));
    const char job = 'S';
    int lwork = -1;
    int info = 0;
    C wq{0, 0};
    auto *lib =
        Pennylane::Util::BLASLibLoaderManager::getInstance().getBLASLib();
    if constexpr (std::is_same_v<fp_t, double>) {
        auto f = lib->getSymbol<zgesvdPtr>("scipy_zgesvd_");
        f(&job, &job, &mm, &nn, A.data(), &lda, S.data(), Ubuf.data(), &ldu,
          VTbuf.data(), &ldvt, &wq, &lwork, rwork.data(), &info);
        lwork = std::max(1, static_cast<int>(wq.real()));
        std::vector<C> work(static_cast<std::size_t>(lwork));
        f(&job, &job, &mm, &nn, A.data(), &lda, S.data(), Ubuf.data(), &ldu,
          VTbuf.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
    } else {
        auto f = lib->getSymbol<cgesvdPtr>("scipy_cgesvd_");
        f(&job, &job, &mm, &nn, A.data(), &lda, S.data(), Ubuf.data(), &ldu,
          VTbuf.data(), &ldvt, &wq, &lwork, rwork.data(), &info);
        lwork = std::max(1, static_cast<int>(wq.real()));
        std::vector<C> work(static_cast<std::size_t>(lwork));
        f(&job, &job, &mm, &nn, A.data(), &lda, S.data(), Ubuf.data(), &ldu,
          VTbuf.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
    }
    if (info != 0) {
        return false;
    }
    out.S = std::move(S);
    out.U = std::move(VTbuf); // row-major m x r
    out.Vh = std::move(Ubuf); // row-major r x n
    return true;
}
} // namespace detail
#endif

/// Whether the SVD uses LAPACK (scipy-openblas)
constexpr bool svdUsesLapack() {
#ifdef SCIPY_OPENBLAS_ENABLED
    return true;
#else
    return false;
#endif
}

/**
 * @brief Thin SVD of a row-major m x n matrix.
 */
template <class fp_t>
SVDResult<fp_t> svd(const std::vector<std::complex<fp_t>> &M, std::size_t m,
                    std::size_t n) {
    PL_ABORT_IF_NOT(M.size() == m * n, "svd: size mismatch");
#ifdef SCIPY_OPENBLAS_ENABLED
    SVDResult<fp_t> out;
    if (detail::lapackSVD<fp_t>(M, m, n, out)) {
        return out;
    }
#endif
    return jacobiSVD<fp_t>(M, m, n);
}

} // namespace Pennylane::LightningTensornet
