#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include "Util.hpp"

namespace Pennylane::Gates {

/**
 * @brief Create a matrix representation of the Identity gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * Identity data.
 */
template <class T>
static constexpr auto getIdentity() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * PauliX data.
 */
template <class T>
static constexpr auto getPauliX() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ZERO<T>(), ONE<T>(), ONE<T>(), ZERO<T>()};
}

/**
 * @brief Create a matrix representation of the PauliY gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * PauliY data.
 */
template <class T>
static constexpr auto getPauliY() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ZERO<T>(), -IMAG<T>(), IMAG<T>(), ZERO<T>()};
}

/**
 * @brief Create a matrix representation of the PauliZ gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * PauliZ data.
 */
template <class T>
static constexpr auto getPauliZ() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), -ONE<T>()};
}

/**
 * @brief Create a matrix representation of the Hadamard gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * Hadamard data.
 */
template <class T>
static constexpr auto getHadamard() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {INVSQRT2<T>(), INVSQRT2<T>(), INVSQRT2<T>(), -INVSQRT2<T>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * S gate data.
 */
template <class T>
static constexpr auto getS() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), IMAG<T>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * T gate data.
 */
template <class T>
static constexpr auto getT() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(), ZERO<T>(), ZERO<T>(),
            std::complex<T>{INVSQRT2<T>(), INVSQRT2<T>()}};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * CNOT gate data.
 */
template <class T>
static constexpr auto getCNOT() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>()};
}

/**
 * @brief Create a matrix representation of the SWAP gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * SWAP gate data.
 */
template <class T>
static constexpr auto getSWAP() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

/**
 * @brief Create a matrix representation of the CZ gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * SWAP gate data.
 */
template <class T>
static constexpr auto getCZ() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), -ONE<T>()};
}

/**
 * @brief Create a matrix representation of the CSWAP gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * CSWAP gate data.
 */
template <class T>
static constexpr auto getCSWAP() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

/**
 * @brief Create a matrix representation of the Toffoli gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * Toffoli gate data.
 */
template <class T>
static constexpr auto getToffoli() -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>()};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const Phase-shift gate
 * data.
 */
template <class T, class U = T>
static auto getPhaseShift(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), std::exp(IMAG<T>() * angle)};
}

/**
 * @brief Create a matrix representation of the Phase-shift gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const Phase-shift gate
 * data.
 */
template <class T, class U = T>
static auto getPhaseShift(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getPhaseShift<T>(params.front());
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RX gate data.
 */
template <class T, class U = T>
static auto getRX(U angle) -> std::vector<std::complex<T>> {
    const std::complex<T> c(std::cos(angle / 2), 0);
    const std::complex<T> js(0, -std::sin(angle / 2));
    return {c, js, js, c};
}

/**
 * @brief Create a matrix representation of the RX gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RX gate data.
 */
template <class T, class U = T>
static auto getRX(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getRX<T>(params.front());
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RY gate data.
 */
template <class T, class U = T>
static auto getRY(U angle) -> std::vector<std::complex<T>> {
    const std::complex<T> c(std::cos(angle / 2), 0);
    const std::complex<T> s(std::sin(angle / 2), 0);
    return {c, -s, s, c};
}

/**
 * @brief Create a matrix representation of the RY gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RY gate data.
 */
template <class T, class U = T>
static auto getRY(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getRY<T>(params.front());
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RZ gate data.
 */
template <class T, class U = T>
static auto getRZ(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    return {std::exp(-IMAG<T>() * (angle / 2)), ZERO<T>(), ZERO<T>(),
            std::exp(IMAG<T>() * (angle / 2))};
}

/**
 * @brief Create a matrix representation of the RZ gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RZ gate data.
 */
template <class T, class U = T>
static auto getRZ(const std::vector<U> &params) -> std::vector<T> {
    return getRZ<T>(params.front());
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
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param phi \f$\phi\f$ shift angle.
 * @param theta \f$\theta\f$ shift angle.
 * @param omega \f$\omega\f$ shift angle.
 * @return const std::vector<std::complex<T>> Return const Rot gate data.
 */
template <class T, class U = T>
static auto getRot(U phi, U theta, U omega) -> std::array<std::complex<T>, 4> {
    using namespace Util;
    const T c = std::cos(theta / 2);
    const T s = std::sin(theta / 2);
    const U p{phi + omega};
    const U m{phi - omega};
    return {std::exp(static_cast<T>(p / 2) * (-IMAG<T>())) * c,
            -std::exp(static_cast<T>(m / 2) * IMAG<T>()) * s,
            std::exp(static_cast<T>(m / 2) * (-IMAG<T>())) * s,
            std::exp(static_cast<T>(p / 2) * IMAG<T>()) * c};
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
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of gate data. Values are expected in order of \f$[\phi,
\theta, \omega]\f$.
 * @return const std::vector<std::complex<T>> Return const Rot gate data.
 */
template <class T, class U = T>
static auto getRot(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getRot<T>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RX gate data.
 */
template <class T, class U = T>
static auto getCRX(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    const std::complex<T> rx{getRX<T>(angle)};
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), rx[0],     rx[1],
            ZERO<T>(), ZERO<T>(), rx[2],     rx[3]};
}

/**
 * @brief Create a matrix representation of the controlled RX gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RX gate data.
 */
template <class T, class U = T>
static auto getCRX(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getCRX<T>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RY gate data.
 */
template <class T, class U = T>
static auto getCRY(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    const std::complex<T> ry{getRY<T>(angle)};
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ry[0],     ry[1],
            ZERO<T>(), ZERO<T>(), ry[2],     ry[3]};
}

/**
 * @brief Create a matrix representation of the controlled RY gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RY gate data.
 */
template <class T, class U = T>
static auto getCRY(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getCRY<T>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param angle Phase shift angle.
 * @return const std::vector<std::complex<T>> Return const RZ gate data.
 */
template <class T, class U = T>
static auto getCRZ(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    const std::complex<T> first = std::exp(-IMAG<T>() * (angle / 2));
    const std::complex<T> second = std::exp(IMAG<T>() * (angle / 2));
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            first,    ZERO<T>(), ZERO<T>(), ZERO<T>(), second};
}

/**
 * @brief Create a matrix representation of the controlled RZ gate data in
 * row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @tparam U Required precision of parameter (`float` or `double`).
 * @param params Vector of phase shift angles. Only front element is read.
 * @return const std::vector<std::complex<T>> Return const RZ gate data.
 */
template <class T, class U = T>
static auto getCRZ(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getCRZ<T>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <class T, class U = T>
static auto getCRot(U phi, U theta, U omega) -> std::vector<std::complex<T>> {
    using namespace Util;
    const std::vector<std::complex<T>> rot{
        std::move(getRot<T>(phi, theta, omega))};
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), rot[0],    rot[1],
            ZERO<T>(), ZERO<T>(), rot[2],    rot[3]};
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(const std::vector<U> &params)`.
 */
template <class T, class U = T>
static auto getCRot(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getCRot<T>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate data
in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <class T, class U = T>
static auto getControlledPhaseShift(U angle) -> std::vector<std::complex<T>> {
    using namespace Util;
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), std::exp(IMAG<T>() * angle)};
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate data
in row-major format.
 *
 * @see `getPhaseShift<T,U>(const std::vector<U> &params)`.
 */
template <class T, class U = T>
static auto getControlledPhaseShift(const std::vector<U> &params)
    -> std::vector<std::complex<T>> {
    return getControlledPhaseShift<T>(params.front());
}

} // namespace Pennylane::Gates
