// Copyright 2018-2023 Xanadu Quantum Technologies Inc.
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
#include <vector>

#include "GateOperation.hpp"
#include "Util.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::Util;
using namespace Pennylane::Gates;
} // namespace
/// @endcond

namespace Pennylane::Gates {

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliX data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getIdentity() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliX data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliX() -> std::vector<ComplexT<T>> {
    return {ZERO<ComplexT, T>(), ONE<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliY gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliY data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliY() -> std::vector<ComplexT<T>> {
    return {ZERO<ComplexT, T>(), -IMAG<ComplexT, T>(), IMAG<ComplexT, T>(),
            ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the PauliZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of PauliZ data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getPauliZ() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            -ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Hadamard gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of Hadamard data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getHadamard() -> std::vector<ComplexT<T>> {
    return {INVSQRT2<ComplexT, T>(), INVSQRT2<ComplexT, T>(),
            INVSQRT2<ComplexT, T>(), -INVSQRT2<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of S gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getS(const bool inverse = false)
    -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            (inverse) ? -IMAG<ComplexT, T>() : IMAG<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of T gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getT(const bool inverse = false)
    -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            INVSQRT2<ComplexT, T>() *
                (ONE<ComplexT, T>() +
                 ((inverse) ? -IMAG<ComplexT, T>() : IMAG<ComplexT, T>()))};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of CNOT gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCNOT() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the SWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getSWAP() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CY gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCY() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), -IMAG<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), IMAG<ComplexT, T>(),
            ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of SWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCZ() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            -ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the CSWAP gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of CSWAP gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getCSWAP() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Toffoli gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>> Return constant expression
 * of Toffoli gate data.
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getToffoli() -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Phase-shift gate
 * data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getPhaseShift(T angle) -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRX(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> c{std::cos(angle / 2), 0};
    const ComplexT<T> js{0, -std::sin(angle / 2)};
    return {c, js, js, c};
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRY(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> c{std::cos(angle / 2), 0};
    const ComplexT<T> s{std::sin(angle / 2), 0};
    return {c, -s, s, c};
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getRZ(T angle) -> std::vector<ComplexT<T>> {
    return {{std::cos(-angle / 2), std::sin(-angle / 2)},
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            {std::cos(angle / 2), std::sin(angle / 2)}};
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
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param phi \f$\phi\f$ shift angle.
 * @param theta \f$\theta\f$ shift angle.
 * @param omega \f$\omega\f$ shift angle.
 * @return std::vector<ComplexT<T>> Return Rot gate data.
 */
template <template <typename...> class ComplexT, typename T, typename U = T>
static auto getRot(U phi, U theta, U omega) -> std::vector<ComplexT<T>> {
    const T c = std::cos(theta / 2);
    const T s = std::sin(theta / 2);
    const U p{phi + omega};
    const U m{phi - omega};
    return {ComplexT<T>{std::cos(p / 2) * c, -std::sin(p / 2) * c},
            ComplexT<T>{-std::cos(m / 2) * s, -std::sin(m / 2) * s},
            ComplexT<T>{std::cos(m / 2) * s, -std::sin(m / 2) * s},
            ComplexT<T>{std::cos(p / 2) * c, std::sin(p / 2) * c}};
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RX gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRX(T angle) -> std::vector<ComplexT<T>> {
    const auto rx{getRX<ComplexT, T>(angle)};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            rx[0],
            rx[1],
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            rx[2],
            rx[3]};
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RY gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRY(T angle) -> std::vector<ComplexT<T>> {
    const auto ry{getRY<ComplexT, T>(angle)};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ry[0],
            ry[1],
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ry[2],
            ry[3]};
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return RZ gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRZ(T angle) -> std::vector<ComplexT<T>> {
    const ComplexT<T> first{std::cos(-angle / 2), std::sin(-angle / 2)};
    const ComplexT<T> second{std::cos(angle / 2), std::sin(angle / 2)};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            first,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            second};
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getCRot(T phi, T theta, T omega) -> std::vector<ComplexT<T>> {
    const auto rot{getRot<ComplexT, T>(phi, theta, omega)};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            rot[0],
            rot[1],
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            rot[2],
            rot[3]};
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate
data in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <template <typename...> class ComplexT, typename T>
static auto getControlledPhaseShift(T angle) -> std::vector<ComplexT<T>> {
    return {ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(), {std::cos(angle), std::sin(angle)}};
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitation(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            c,
            -s,
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            s,
            c,
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the SingleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitation()
    -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        IMAG<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), -IMAG<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationMinus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = exp(ComplexT<T>(0, -p2));
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    return {e,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            c,
            -s,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            s,
            c,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            e};
}

/**
 * @brief Create a matrix representation of the SingleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitationMinus()
    -> std::vector<ComplexT<T>> {
    return {
        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        IMAG<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), -IMAG<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return single excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getSingleExcitationPlus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = exp(ComplexT<T>(0, p2));
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    return {e,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            c,
            -s,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            s,
            c,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            e};
}

/**
 * @brief Create a matrix representation of the SingleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorSingleExcitationPlus()
    -> std::vector<ComplexT<T>> {
    return {
        -ONE<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        IMAG<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), -IMAG<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), -ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitation(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
    mat[0] = ONE<ComplexT, T>();
    mat[17] = ONE<ComplexT, T>();
    mat[34] = ONE<ComplexT, T>();
    mat[51] = c;
    mat[60] = -s;
    mat[68] = ONE<ComplexT, T>();
    mat[85] = ONE<ComplexT, T>();
    mat[102] = ONE<ComplexT, T>();
    mat[119] = ONE<ComplexT, T>();
    mat[136] = ONE<ComplexT, T>();
    mat[153] = ONE<ComplexT, T>();
    mat[170] = ONE<ComplexT, T>();
    mat[187] = ONE<ComplexT, T>();
    mat[195] = s;
    mat[204] = c;
    mat[221] = ONE<ComplexT, T>();
    mat[238] = ONE<ComplexT, T>();
    mat[255] = ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the DoubleExcitation
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitation()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
    mat[60] = IMAG<ComplexT, T>();
    mat[195] = -IMAG<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with negative phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationMinus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = exp(ComplexT<T>(0, -p2));
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
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
 * @brief Create a matrix representation of the DoubleExcitation Minus
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitationMinus()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
    mat[0] = ONE<ComplexT, T>();
    mat[17] = ONE<ComplexT, T>();
    mat[34] = ONE<ComplexT, T>();
    mat[60] = IMAG<ComplexT, T>();
    mat[68] = ONE<ComplexT, T>();
    mat[85] = ONE<ComplexT, T>();
    mat[102] = ONE<ComplexT, T>();
    mat[119] = ONE<ComplexT, T>();
    mat[136] = ONE<ComplexT, T>();
    mat[153] = ONE<ComplexT, T>();
    mat[170] = ONE<ComplexT, T>();
    mat[187] = ONE<ComplexT, T>();
    mat[195] = -IMAG<ComplexT, T>();
    mat[221] = ONE<ComplexT, T>();
    mat[238] = ONE<ComplexT, T>();
    mat[255] = ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data in
 * row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return double excitation rotation
 * with positive phase-shift outside the rotation subspace gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getDoubleExcitationPlus(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> e = exp(ComplexT<T>(0, p2));
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> s{std::sin(p2), 0};
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
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
 * @brief Create a matrix representation of the DoubleExcitation Plus
 * generator data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorDoubleExcitationPlus()
    -> std::vector<ComplexT<T>> {
    std::vector<ComplexT<T>> mat(256, ZERO<ComplexT, T>());
    mat[0] = -ONE<ComplexT, T>();
    mat[17] = -ONE<ComplexT, T>();
    mat[34] = -ONE<ComplexT, T>();
    mat[60] = IMAG<ComplexT, T>();
    mat[68] = -ONE<ComplexT, T>();
    mat[85] = -ONE<ComplexT, T>();
    mat[102] = -ONE<ComplexT, T>();
    mat[119] = -ONE<ComplexT, T>();
    mat[136] = -ONE<ComplexT, T>();
    mat[153] = -ONE<ComplexT, T>();
    mat[170] = -ONE<ComplexT, T>();
    mat[187] = -ONE<ComplexT, T>();
    mat[195] = -IMAG<ComplexT, T>();
    mat[221] = -ONE<ComplexT, T>();
    mat[238] = -ONE<ComplexT, T>();
    mat[255] = -ONE<ComplexT, T>();
    return mat;
}

/**
 * @brief Create a matrix representation of the Ising XX coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising XX coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingXX(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> neg_is{0, -std::sin(p2)};
    return {c,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            neg_is,
            ZERO<ComplexT, T>(),
            c,
            neg_is,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            neg_is,
            c,
            ZERO<ComplexT, T>(),
            neg_is,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorPhaseShift() -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(),
        ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorControlledPhaseShift()
    -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising XX generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingXX() -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising XY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising XY coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingXY(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> neg_is{0, std::sin(p2)};
    return {ONE<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            c,
            neg_is,
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            neg_is,
            c,
            ZERO<ComplexT, T>(),

            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ONE<ComplexT, T>()};
}

/**
 * @brief Create a matrix representation of the Ising XY generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingXY() -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising YY coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising YY coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingYY(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> c{std::cos(p2), 0};
    const ComplexT<T> pos_is{0, std::sin(p2)};
    const ComplexT<T> neg_is{0, -std::sin(p2)};
    return {c,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            pos_is,
            ZERO<ComplexT, T>(),
            c,
            neg_is,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            neg_is,
            c,
            ZERO<ComplexT, T>(),
            pos_is,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            c};
}

/**
 * @brief Create a matrix representation of the Ising YY generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::array<CFP_t>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingYY() -> std::vector<ComplexT<T>> {
    return {
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), -ONE<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        -ONE<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
    };
}

/**
 * @brief Create a matrix representation of the Ising ZZ coupling
 * gate data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return std::vector<ComplexT<T>> Return Ising ZZ coupling
 * gate data.
 */
template <template <typename...> class ComplexT, typename T>
static auto getIsingZZ(T angle) -> std::vector<ComplexT<T>> {
    const T p2 = angle / 2;
    const ComplexT<T> neg_e = exp(ComplexT<T>(0, -p2));
    const ComplexT<T> pos_e = exp(ComplexT<T>(0, p2));
    return {neg_e,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            pos_e,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            pos_e,
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            ZERO<ComplexT, T>(),
            neg_e};
}

/**
 * @brief Create a matrix representation of the Ising ZZ generator
 * data in row-major format.
 *
 * @tparam ComplexT Complex class.
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam T Required precision of parameter (`float` or `double`).
 * @return constexpr std::vector<ComplexT<T>>
 */
template <template <typename...> class ComplexT, typename T>
static constexpr auto getGeneratorIsingZZ() -> std::vector<ComplexT<T>> {
    return {
        -ONE<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ONE<ComplexT, T>(),
        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ONE<ComplexT, T>(),  ZERO<ComplexT, T>(),

        ZERO<ComplexT, T>(), ZERO<ComplexT, T>(),
        ZERO<ComplexT, T>(), -ONE<ComplexT, T>(),
    };
}

template <template <typename...> class ComplexT, typename T>
std::vector<ComplexT<T>> getMatrix(const GateOperation gate_op,
                                   const std::vector<T> &params,
                                   const bool inverse = false) {
    switch (gate_op) {
    case GateOperation::Identity:
        return getIdentity<ComplexT, T>();
    case GateOperation::PauliX:
        return getPauliX<ComplexT, T>();
    case GateOperation::PauliY:
        return getPauliY<ComplexT, T>();
    case GateOperation::PauliZ:
        return getPauliZ<ComplexT, T>();
    case GateOperation::Hadamard:
        return getHadamard<ComplexT, T>();
    case GateOperation::S:
        return getS<ComplexT, T>(inverse);
    case GateOperation::T:
        return getT<ComplexT, T>(inverse);
    case GateOperation::RX:
        return getRX<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::RY:
        return getRY<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::RZ:
        return getRZ<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::PhaseShift:
        return getPhaseShift<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::Rot:
        return (inverse)
                   ? getRot<ComplexT, T>(-params[2], -params[1], -params[0])
                   : getRot<ComplexT, T>(params[0], params[1], params[2]);
    case GateOperation::CNOT:
        return getCNOT<ComplexT, T>();
    case GateOperation::CY:
        return getCY<ComplexT, T>();
    case GateOperation::CZ:
        return getCZ<ComplexT, T>();
    case GateOperation::SWAP:
        return getSWAP<ComplexT, T>();
    case GateOperation::ControlledPhaseShift:
        return getControlledPhaseShift<ComplexT, T>((inverse) ? -params[0]
                                                              : params[0]);
    case GateOperation::CRX:
        return getCRX<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::CRY:
        return getCRY<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::CRZ:
        return getCRZ<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::CRot:
        return (inverse)
                   ? getCRot<ComplexT, T>(-params[2], -params[1], -params[0])
                   : getCRot<ComplexT, T>(params[0], params[1], params[2]);
    case GateOperation::IsingXX:
        return getIsingXX<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::IsingXY:
        return getIsingXY<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::IsingYY:
        return getIsingYY<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::IsingZZ:
        return getIsingZZ<ComplexT, T>((inverse) ? -params[0] : params[0]);
    case GateOperation::SingleExcitation:
        return getSingleExcitation<ComplexT, T>((inverse) ? -params[0]
                                                          : params[0]);
    case GateOperation::SingleExcitationMinus:
        return getSingleExcitationMinus<ComplexT, T>((inverse) ? -params[0]
                                                               : params[0]);
    case GateOperation::SingleExcitationPlus:
        return getSingleExcitationPlus<ComplexT, T>((inverse) ? -params[0]
                                                              : params[0]);
    case GateOperation::DoubleExcitation:
        return getDoubleExcitation<ComplexT, T>((inverse) ? -params[0]
                                                          : params[0]);
    case GateOperation::DoubleExcitationMinus:
        return getDoubleExcitationMinus<ComplexT, T>((inverse) ? -params[0]
                                                               : params[0]);
    case GateOperation::DoubleExcitationPlus:
        return getDoubleExcitationPlus<ComplexT, T>((inverse) ? -params[0]
                                                              : params[0]);
    case GateOperation::CSWAP:
        return getCSWAP<ComplexT, T>();
    case GateOperation::Toffoli:
        return getToffoli<ComplexT, T>();
    default:
        PL_ABORT("This GateOperation does not have a corresponding matrix.");
    }
}

template <template <typename...> class ComplexT, typename T>
std::vector<ComplexT<T>>
getGeneratorMatrix(const GeneratorOperation generator_op) {
    switch (generator_op) {
    case GeneratorOperation::RX:
        return getPauliX<ComplexT, T>();
    case GeneratorOperation::RY:
        return getPauliY<ComplexT, T>();
    case GeneratorOperation::RZ:
        return getPauliZ<ComplexT, T>();
    case GeneratorOperation::PhaseShift:
        return getGeneratorPhaseShift<ComplexT, T>();
    case GeneratorOperation::ControlledPhaseShift:
        return getGeneratorControlledPhaseShift<ComplexT, T>();
    case GeneratorOperation::CRX:
        return getCNOT<ComplexT, T>();
    case GeneratorOperation::CRY:
        return getCY<ComplexT, T>();
    case GeneratorOperation::CRZ:
        return getCZ<ComplexT, T>();
    case GeneratorOperation::IsingXX:
        return getGeneratorIsingXX<ComplexT, T>();
    case GeneratorOperation::IsingXY:
        return getGeneratorIsingXY<ComplexT, T>();
    case GeneratorOperation::IsingYY:
        return getGeneratorIsingYY<ComplexT, T>();
    case GeneratorOperation::IsingZZ:
        return getGeneratorIsingZZ<ComplexT, T>();
    case GeneratorOperation::SingleExcitation:
        return getGeneratorSingleExcitation<ComplexT, T>();
    case GeneratorOperation::SingleExcitationMinus:
        return getGeneratorSingleExcitationMinus<ComplexT, T>();
    case GeneratorOperation::SingleExcitationPlus:
        return getGeneratorSingleExcitationPlus<ComplexT, T>();
    case GeneratorOperation::DoubleExcitation:
        return getGeneratorDoubleExcitation<ComplexT, T>();
    case GeneratorOperation::DoubleExcitationMinus:
        return getGeneratorDoubleExcitationMinus<ComplexT, T>();
    case GeneratorOperation::DoubleExcitationPlus:
        return getGeneratorDoubleExcitationPlus<ComplexT, T>();
    default:
        PL_ABORT("This GateOperation does not have a corresponding matrix.");
    }
}

} // namespace Pennylane::Gates
