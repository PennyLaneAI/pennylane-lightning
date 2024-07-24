// Copyright 2022-2023 Xanadu Quantum Technologies Inc.
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

#include <cmath>
#include <complex>
#include <functional>
#include <vector>

#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace cuUtil;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::cuGates {

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of PauliX data.
 */
template <class ComplexT>
static constexpr auto getIdentity() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of PauliX data.
 */
template <class ComplexT>
static constexpr auto getPauliX() -> std::vector<ComplexT> {
    return {cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the PauliY gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of PauliY data.
 */
template <class ComplexT>
static constexpr auto getPauliY() -> std::vector<ComplexT> {
    return {cuUtil::ZERO<ComplexT>(), -cuUtil::IMAG<ComplexT>(),
            cuUtil::IMAG<ComplexT>(), cuUtil::ZERO<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the PauliZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of PauliZ data.
 */
template <class ComplexT>
static constexpr auto getPauliZ() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), -cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the Hadamard gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of Hadamard data.
 */
template <class ComplexT>
static constexpr auto getHadamard() -> std::vector<ComplexT> {
    return {cuUtil::INVSQRT2<ComplexT>(), cuUtil::INVSQRT2<ComplexT>(),
            cuUtil::INVSQRT2<ComplexT>(), -cuUtil::INVSQRT2<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of S gate data.
 */
template <class ComplexT>
static constexpr auto getS() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::IMAG<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of T gate data.
 */
template <class ComplexT>
static constexpr auto getT() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ConstMultSC(
                cuUtil::SQRT2<decltype(cuUtil::ONE<ComplexT>().x)>() / 2,
                cuUtil::ConstSum(cuUtil::ONE<ComplexT>(),
                                 cuUtil::IMAG<ComplexT>()))};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of CNOT gate data.
 */
template <class ComplexT>
static constexpr auto getCNOT() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the SWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of SWAP gate data.
 */
template <class ComplexT>
static constexpr auto getSWAP() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of SWAP gate data.
 */
template <class ComplexT>
static constexpr auto getCY() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), -cuUtil::IMAG<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::IMAG<ComplexT>(), cuUtil::ZERO<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of SWAP gate data.
 */
template <class ComplexT>
static constexpr auto getCZ() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), -cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the CSWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of CSWAP gate data.
 */
template <class ComplexT>
static constexpr auto getCSWAP() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the Toffoli gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT> Return constant expression
 * of Toffoli gate data.
 */
template <class ComplexT>
static constexpr auto getToffoli() -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return Phase-shift gate
 * data.
 */
template <class ComplexT, class U = double>
static auto getPhaseShift(U angle) -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return Phase-shift gate
 * data.
 */
template <class ComplexT, class U = double>
static auto getPhaseShift(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getPhaseShift<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RX gate data.
 */
template <class ComplexT, class U = double>
static auto getRX(U angle) -> std::vector<ComplexT> {
    const ComplexT c{std::cos(angle / 2), 0};
    const ComplexT js{0, -std::sin(angle / 2)};
    return {c, js, js, c};
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RX gate data.
 */
template <class ComplexT, class U = double>
static auto getRX(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getRX<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RY gate data.
 */
template <class ComplexT, class U = double>
static auto getRY(U angle) -> std::vector<ComplexT> {
    const ComplexT c{std::cos(angle / 2), 0};
    const ComplexT s{std::sin(angle / 2), 0};
    return {c, -s, s, c};
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RY gate data.
 */
template <class ComplexT, class U = double>
static auto getRY(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getRY<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RZ gate data.
 */
template <class ComplexT, class U = double>
static auto getRZ(U angle) -> std::vector<ComplexT> {
    return {{std::cos(-angle / 2), std::sin(-angle / 2)},
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            {std::cos(angle / 2), std::sin(angle / 2)}};
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RZ gate data.
 */
template <class ComplexT, class U = double>
static auto getRZ(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getRZ<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param phi \f$\phi\f$ shift angle.
 * @param theta \f$\theta\f$ shift angle.
 * @param omega \f$\omega\f$ shift angle.
 * @return std::vector<ComplexT> Return Rot gate data.
 */
template <class ComplexT, class U = double>
static auto getRot(U phi, U theta, U omega) -> std::vector<ComplexT> {
    const U c = std::cos(theta / 2);
    const U s = std::sin(theta / 2);
    const U p{phi + omega};
    const U m{phi - omega};
    /*
        return {ComplexT{std::cos(p / 2), -std::sin(p / 2)} * c,
                -ComplexT{std::cos(m / 2), std::sin(m / 2)} * s,
                ComplexT{std::cos(m / 2), -std::sin(m / 2)} * s,
                ComplexT{std::cos(p / 2), std::sin(p / 2)} * c};*/
    return {ConstMultSC(c, ComplexT{std::cos(p / 2), -std::sin(p / 2)}),
            ConstMultSC(s, -ComplexT{std::cos(m / 2), std::sin(m / 2)}),
            ConstMultSC(s, ComplexT{std::cos(m / 2), -std::sin(m / 2)}),
            ConstMultSC(c, ComplexT{std::cos(p / 2), std::sin(p / 2)})};
}

/**
 * @brief Create a matrix representation of the Rot gate data in row-major
format.
 *
 * The gate is defined as:
 * \f$\begin{split}Rot(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)=
\begin{bmatrix}
e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
\end{bmatrix}.\end{split}\f$
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of gate data. Values are expected in order of
\f$[\phi, \theta, \omega]\f$.
 * @return std::vector<ComplexT> Return Rot gate data.
 */
template <class ComplexT, class U = double>
static auto getRot(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getRot<ComplexT>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RX gate data.
 */
template <class ComplexT, class U = double>
static auto getCRX(U angle) -> std::vector<ComplexT> {
    const auto rx{getRX<ComplexT>(angle)};
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            rx[0],
            rx[1],
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            rx[2],
            rx[3]};
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RX gate data.
 */
template <class ComplexT, class U = double>
static auto getCRX(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getCRX<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RY gate data.
 */
template <class ComplexT, class U = double>
static auto getCRY(U angle) -> std::vector<ComplexT> {
    const auto ry{getRY<ComplexT>(angle)};
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            ry[0],
            ry[1],
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            ry[2],
            ry[3]};
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RY gate data.
 */
template <class ComplexT, class U = double>
static auto getCRY(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getCRY<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return RZ gate data.
 */
template <class ComplexT, class U = double>
static auto getCRZ(U angle) -> std::vector<ComplexT> {
    const ComplexT first{std::cos(-angle / 2), std::sin(-angle / 2)};
    const ComplexT second{std::cos(angle / 2), std::sin(angle / 2)};
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            first,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            second};
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return RZ gate data.
 */
template <class ComplexT, class U = double>
static auto getCRZ(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getCRZ<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <class ComplexT, class U = double>
static auto getCRot(U phi, U theta, U omega) -> std::vector<ComplexT> {
    const auto rot{std::move(getRot<ComplexT>(phi, theta, omega))};
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            rot[0],
            rot[1],
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            rot[2],
            rot[3]};
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(const std::vector<U> &params)`.
 */
template <class ComplexT, class U = double>
static auto getCRot(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getCRot<ComplexT>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <class ComplexT, class U = double>
static auto getControlledPhaseShift(U angle) -> std::vector<ComplexT> {
    return {cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(const std::vector<U> &params)`.
 */
template <class ComplexT, class U = double>
static auto getControlledPhaseShift(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getControlledPhaseShift<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return single excitation rotation
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitation(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT c{std::cos(p2), 0};
    // TODO: To remove conditional compilation here in the future, current
    // implementation will block the simultaneous installation of LGPU and
    // cutensornet backends

#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c,
            -s,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            s,
            c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return single excitation rotation
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitation(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getSingleExcitation<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorSingleExcitation() -> std::vector<ComplexT> {
    return {
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::IMAG<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), -cuUtil::IMAG<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitationMinus(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const ComplexT c{std::cos(p2), 0};
// TODO: To remove conditional compilation here in the future, current
// implementation will block the simultaneous installation of LGPU and
// cutensornet backends
#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif

    return {e,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c,
            -s,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            s,
            c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitationMinus(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getSingleExcitationMinus<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorSingleExcitationMinus()
    -> std::vector<ComplexT> {
    return {
        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::IMAG<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), -cuUtil::IMAG<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitationPlus(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    const ComplexT c{std::cos(p2), 0};
    // TODO: To remove conditional compilation here in the future, current
    // implementation will block the simultaneous installation of LGPU and
    // cutensornet backends

#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif
    return {e,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c,
            -s,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            s,
            c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            e};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getSingleExcitationPlus(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getSingleExcitationPlus<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the SingleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorSingleExcitationPlus()
    -> std::vector<ComplexT> {
    return {
        -cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::IMAG<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), -cuUtil::IMAG<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), -cuUtil::ONE<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return double excitation rotation
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitation(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT c{std::cos(p2), 0};
    // TODO: To remove conditional compilation here in the future, current
    // implementation will block the simultaneous installation of LGPU and
    // cutensornet backends

#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[0] = cuUtil::ONE<ComplexT>();
    mat[17] = cuUtil::ONE<ComplexT>();
    mat[34] = cuUtil::ONE<ComplexT>();
    mat[51] = c;
    mat[60] = -s;
    mat[68] = cuUtil::ONE<ComplexT>();
    mat[85] = cuUtil::ONE<ComplexT>();
    mat[102] = cuUtil::ONE<ComplexT>();
    mat[119] = cuUtil::ONE<ComplexT>();
    mat[136] = cuUtil::ONE<ComplexT>();
    mat[153] = cuUtil::ONE<ComplexT>();
    mat[170] = cuUtil::ONE<ComplexT>();
    mat[187] = cuUtil::ONE<ComplexT>();
    mat[195] = s;
    mat[204] = c;
    mat[221] = cuUtil::ONE<ComplexT>();
    mat[238] = cuUtil::ONE<ComplexT>();
    mat[255] = cuUtil::ONE<ComplexT>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return double excitation rotation
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitation(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getDoubleExcitation<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorDoubleExcitation() -> std::vector<ComplexT> {
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[60] = cuUtil::IMAG<ComplexT>();
    mat[195] = -cuUtil::IMAG<ComplexT>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitationMinus(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const ComplexT c{std::cos(p2), 0};
    // TODO: To remove conditional compilation here in the future, current
    // implementation will block the simultaneous installation of LGPU and
    // cutensornet backends

#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = -s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitationMinus(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getDoubleExcitationMinus<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorDoubleExcitationMinus()
    -> std::vector<ComplexT> {
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[0] = cuUtil::ONE<ComplexT>();
    mat[17] = cuUtil::ONE<ComplexT>();
    mat[34] = cuUtil::ONE<ComplexT>();
    mat[60] = cuUtil::IMAG<ComplexT>();
    mat[68] = cuUtil::ONE<ComplexT>();
    mat[85] = cuUtil::ONE<ComplexT>();
    mat[102] = cuUtil::ONE<ComplexT>();
    mat[119] = cuUtil::ONE<ComplexT>();
    mat[136] = cuUtil::ONE<ComplexT>();
    mat[153] = cuUtil::ONE<ComplexT>();
    mat[170] = cuUtil::ONE<ComplexT>();
    mat[187] = cuUtil::ONE<ComplexT>();
    mat[195] = -cuUtil::IMAG<ComplexT>();
    mat[221] = cuUtil::ONE<ComplexT>();
    mat[238] = cuUtil::ONE<ComplexT>();
    mat[255] = cuUtil::ONE<ComplexT>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitationPlus(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    const ComplexT c{std::cos(p2), 0};
    // TODO: To remove conditional compilation here in the future, current
    // implementation will block the simultaneous installation of LGPU and
    // cutensornet backends
#ifdef _ENABLE_PLGPU
    const ComplexT s{-std::sin(p2), 0}; // column-major
#else
    const ComplexT s{std::sin(p2), 0}; // row-major
#endif
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[0] = e;
    mat[17] = e;
    mat[34] = e;
    mat[51] = c;
    mat[60] = -s;
    mat[68] = e;
    mat[85] = e;
    mat[102] = e;
    mat[119] = e;
    mat[136] = e;
    mat[153] = e;
    mat[170] = e;
    mat[187] = e;
    mat[195] = s;
    mat[204] = c;
    mat[221] = e;
    mat[238] = e;
    mat[255] = e;
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <class ComplexT, class U = double>
static auto getDoubleExcitationPlus(const std::vector<U> &params)
    -> std::vector<ComplexT> {
    return getDoubleExcitationPlus<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the DoubleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorDoubleExcitationPlus()
    -> std::vector<ComplexT> {
    std::vector<ComplexT> mat(256, cuUtil::ZERO<ComplexT>());
    mat[0] = -cuUtil::ONE<ComplexT>();
    mat[17] = -cuUtil::ONE<ComplexT>();
    mat[34] = -cuUtil::ONE<ComplexT>();
    mat[60] = cuUtil::IMAG<ComplexT>();
    mat[68] = -cuUtil::ONE<ComplexT>();
    mat[85] = -cuUtil::ONE<ComplexT>();
    mat[102] = -cuUtil::ONE<ComplexT>();
    mat[119] = -cuUtil::ONE<ComplexT>();
    mat[136] = -cuUtil::ONE<ComplexT>();
    mat[153] = -cuUtil::ONE<ComplexT>();
    mat[170] = -cuUtil::ONE<ComplexT>();
    mat[187] = -cuUtil::ONE<ComplexT>();
    mat[195] = -cuUtil::IMAG<ComplexT>();
    mat[221] = -cuUtil::ONE<ComplexT>();
    mat[238] = -cuUtil::ONE<ComplexT>();
    mat[255] = -cuUtil::ONE<ComplexT>();
    return mat;
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return Ising XX coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingXX(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT c{std::cos(p2), 0};
    const ComplexT neg_is{0, -std::sin(p2)};
    return {c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            neg_is,
            cuUtil::ZERO<ComplexT>(),
            c,
            neg_is,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            neg_is,
            c,
            cuUtil::ZERO<ComplexT>(),
            neg_is,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return Ising XX coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingXX(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getIsingXX<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorIsingXX() -> std::vector<ComplexT> {
    return {
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return Ising YY coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingYY(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT c{std::cos(p2), 0};
    const ComplexT pos_is{0, std::sin(p2)};
    const ComplexT neg_is{0, -std::sin(p2)};
    return {c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            pos_is,
            cuUtil::ZERO<ComplexT>(),
            c,
            neg_is,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            neg_is,
            c,
            cuUtil::ZERO<ComplexT>(),
            pos_is,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return Ising YY coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingYY(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getIsingYY<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising YY generator
 * data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorIsingYY() -> std::vector<ComplexT> {
    return {
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), -cuUtil::ONE<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        -cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return Ising ZZ coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingZZ(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT neg_e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, -p2)));
    const ComplexT pos_e =
        cuUtil::complexToCu<std::complex<U>>(std::exp(std::complex<U>(0, p2)));
    return {neg_e,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            pos_e,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            pos_e,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            neg_e};
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return Ising ZZ coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingZZ(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getIsingZZ<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising ZZ generator
 * data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorIsingZZ() -> std::vector<ComplexT> {
    return {
        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), -cuUtil::ONE<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        -cuUtil::ONE<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising XY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT> Return Ising XY coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingXY(U angle) -> std::vector<ComplexT> {
    const U p2 = angle / 2;
    const ComplexT c{std::cos(p2), 0};
    const ComplexT pos_is{0, std::sin(p2)};
    return {cuUtil::ONE<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            c,
            pos_is,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            pos_is,
            c,
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(),
            cuUtil::ONE<ComplexT>()};
}

/**
 * @brief Create a matrix representation of the Ising XY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return std::vector<ComplexT> Return Ising XY coupling
 * gate data.
 */
template <class ComplexT, class U = double>
static auto getIsingXY(const std::vector<U> &params) -> std::vector<ComplexT> {
    return getIsingXY<ComplexT>(params.front());
}

/**
 * @brief Create a matrix representation of the Ising XY generator
 * data in row-major format.
 *
 * @tparam ComplexT Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<ComplexT>
 */
template <class ComplexT, class U = double>
static constexpr auto getGeneratorIsingXY() -> std::vector<ComplexT> {
    return {
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ONE<ComplexT>(),  cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),

        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
        cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
    };
}

template <class ComplexT>
static constexpr auto getP11_CU() -> std::vector<ComplexT> {
    return {cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>()};
}

template <class ComplexT>
static constexpr auto getP1111_CU() -> std::vector<ComplexT> {
    return {cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ZERO<ComplexT>(),
            cuUtil::ZERO<ComplexT>(), cuUtil::ONE<ComplexT>()};
}

/*
 * @brief Dyanmical access the gate data based on the gate name and parameters.
 *
 * @tparam PrecisionT Required precision of gate (`float` or `double`).
 */
template <class PrecisionT> class DynamicGateDataAccess {
  private:
    DynamicGateDataAccess() = default;

  public:
    using ComplexT = decltype(cuUtil::getCudaType(PrecisionT{}));
    DynamicGateDataAccess(DynamicGateDataAccess &&) = delete;
    DynamicGateDataAccess(const DynamicGateDataAccess &) = delete;
    DynamicGateDataAccess &operator=(const DynamicGateDataAccess &) = delete;

    ~DynamicGateDataAccess() = default;

  public:
    static DynamicGateDataAccess &getInstance() {
        static DynamicGateDataAccess instance;
        return instance;
    }

    auto
    getGateData(const std::string &gate_name,
                [[maybe_unused]] const std::vector<PrecisionT> &params) const
        -> std::vector<ComplexT> {
        if (nonparametric_gates_.find(gate_name) !=
            nonparametric_gates_.end()) {
            return nonparametric_gates_.at(gate_name)();
        } else if (parametric_gates_.find(gate_name) !=
                   parametric_gates_.end()) {
            return parametric_gates_.at(gate_name)(params);
        } else {
            throw std::invalid_argument("Unsupported gate: " + gate_name + ".");
        }
    }

  private:
    using ParamGateFunc =
        std::function<std::vector<ComplexT>(const std::vector<PrecisionT> &)>;
    using NonParamGateFunc = std::function<std::vector<ComplexT>()>;
    using ParamGateFuncMap = std::unordered_map<std::string, ParamGateFunc>;
    using NonParamGateFuncMap =
        std::unordered_map<std::string, NonParamGateFunc>;

    // TODO: Need changes to support to the controlled gate tensor API once the
    // API is finalized in cutensornet lib.
    NonParamGateFuncMap nonparametric_gates_{
        {"Identity",
         []() -> std::vector<ComplexT> {
             return cuGates::getIdentity<ComplexT>();
         }},
        {"PauliX",
         []() -> std::vector<ComplexT> {
             return cuGates::getPauliX<ComplexT>();
         }},
        {"PauliY",
         []() -> std::vector<ComplexT> {
             return cuGates::getPauliY<ComplexT>();
         }},
        {"PauliZ",
         []() -> std::vector<ComplexT> {
             return cuGates::getPauliZ<ComplexT>();
         }},
        {"S",
         []() -> std::vector<ComplexT> { return cuGates::getS<ComplexT>(); }},
        {"Hadamard",
         []() -> std::vector<ComplexT> {
             return cuGates::getHadamard<ComplexT>();
         }},
        {"T",
         []() -> std::vector<ComplexT> { return cuGates::getT<ComplexT>(); }},
        {"SWAP",
         []() -> std::vector<ComplexT> {
             return cuGates::getSWAP<ComplexT>();
         }},
        {"CNOT",
         []() -> std::vector<ComplexT> {
             return cuGates::getCNOT<ComplexT>();
         }},
        {"Toffoli",
         []() -> std::vector<ComplexT> {
             return cuGates::getToffoli<ComplexT>();
         }},
        {"CY",
         []() -> std::vector<ComplexT> { return cuGates::getCY<ComplexT>(); }},
        {"CZ",
         []() -> std::vector<ComplexT> { return cuGates::getCZ<ComplexT>(); }},
        {"CSWAP", []() -> std::vector<ComplexT> {
             return cuGates::getCSWAP<ComplexT>();
         }}};

    // TODO: Need changes to support to the controlled gate tensor API once the
    // API is finalized in cutensornet lib.
    ParamGateFuncMap parametric_gates_{
        {"PhaseShift",
         [](auto &&params) {
             return cuGates::getPhaseShift<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RX",
         [](auto &&params) {
             return cuGates::getRX<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [](auto &&params) {
             return cuGates::getRY<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [](auto &&params) {
             return cuGates::getRZ<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"Rot",
         [](auto &&params) {
             return cuGates::getRot<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]),
                 std::forward<decltype(params[1])>(params[1]),
                 std::forward<decltype(params[2])>(params[2]));
         }},
        {"CRX",
         [](auto &&params) {
             return cuGates::getCRX<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [](auto &&params) {
             return cuGates::getCRY<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [](auto &&params) {
             return cuGates::getCRZ<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRot",
         [](auto &&params) {
             return cuGates::getCRot<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]),
                 std::forward<decltype(params[1])>(params[1]),
                 std::forward<decltype(params[2])>(params[2]));
         }},
        {"ControlledPhaseShift",
         [](auto &&params) {
             return cuGates::getControlledPhaseShift<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [](auto &&params) {
             return cuGates::getIsingXX<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingYY",
         [](auto &&params) {
             return cuGates::getIsingYY<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingZZ",
         [](auto &&params) {
             return cuGates::getIsingZZ<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXY",
         [](auto &&params) {
             return cuGates::getIsingXY<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitation",
         [](auto &&params) {
             return cuGates::getSingleExcitation<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [](auto &&params) {
             return cuGates::getSingleExcitationMinus<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [](auto &&params) {
             return cuGates::getSingleExcitationPlus<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [](auto &&params) {
             return cuGates::getDoubleExcitation<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [](auto &&params) {
             return cuGates::getDoubleExcitationMinus<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus", [](auto &&params) {
             return cuGates::getDoubleExcitationPlus<ComplexT>(
                 std::forward<decltype(params[0])>(params[0]));
         }}};
};

} // namespace Pennylane::LightningGPU::cuGates
