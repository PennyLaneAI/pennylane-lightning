#pragma once

#include <cmath>
#include <complex>
#include <vector>

#include "Util.hpp"

namespace {
using namespace Pennylane::Util;
}

namespace Pennylane {
namespace Gates {

/**
 * @brief Create a matrix representation of the PauliX gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * PauliX data.
 */
template <class T> static constexpr std::vector<std::complex<T>> getPauliX() {
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
template <class T> static constexpr std::vector<std::complex<T>> getPauliY() {
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
template <class T> static constexpr std::vector<std::complex<T>> getPauliZ() {
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
template <class T> static constexpr std::vector<std::complex<T>> getHadamard() {
    return {INVSQRT2<T>(), INVSQRT2<T>(), INVSQRT2<T>(), -INVSQRT2<T>()};
}

/**
 * @brief Create a matrix representation of the S gate data in row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * S gate data.
 */
template <class T> static constexpr std::vector<std::complex<T>> getS() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), IMAG<T>()};
}

/**
 * @brief Create a matrix representation of the T gate data in row-major format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * T gate data.
 */
template <class T> static constexpr std::vector<std::complex<T>> getT() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), IMAG<T>()};
}

/**
 * @brief Create a matrix representation of the CNOT gate data in row-major
 * format.
 *
 * @tparam T Required precision of gate (`float` or `double`).
 * @return constexpr std::vector<std::complex<T>> Return constant expression of
 * CNOT gate data.
 */
template <class T> static constexpr std::vector<std::complex<T>> getCNOT() {
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
template <class T> static constexpr std::vector<std::complex<T>> getSWAP() {
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
template <class T> static constexpr std::vector<std::complex<T>> getCZ() {
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
template <class T> static constexpr std::vector<std::complex<T>> getCSWAP() {
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
template <class T> static constexpr std::vector<std::complex<T>> getToffoli() {
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
static const std::vector<std::complex<T>> getPhaseShift(U angle) {
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
static const std::vector<std::complex<T>>
getPhaseShift(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getRX(U angle) {
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
static const std::vector<std::complex<T>> getRX(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getRY(U angle) {
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
static const std::vector<std::complex<T>> getRY(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getRZ(U angle) {
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
static const std::vector<T> getRZ(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getRot(U phi, U theta, U omega) {
    const std::complex<T> c(std::cos(theta / 2), 0), s(std::sin(theta / 2), 0);
    const U p{phi + omega}, m{phi - omega};
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
static const std::vector<std::complex<T>> getRot(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getCRX(U angle) {
    const std::complex<T> rx{RX<T>(angle)};
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
static const std::vector<std::complex<T>> getCRX(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getCRY(U angle) {
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
static const std::vector<std::complex<T>> getCRY(const std::vector<U> &params) {
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
static const std::vector<std::complex<T>> getCRZ(U angle) {
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
static const std::vector<std::complex<T>> getCRZ(const std::vector<U> &params) {
    return getCRZ<T>(params.front());
}

/**
 * @brief Create a matrix representation of the controlled Rot gate data in
row-major format.
 *
 * @see `getRot<T,U>(U phi, U theta, U omega)`.
 */
template <class T, class U = T>
static const std::vector<std::complex<T>> getCRot(U phi, U theta, U omega) {
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
static const std::vector<std::complex<T>>
getCRot(const std::vector<U> &params) {
    return getCRot<T>(params[0], params[1], params[2]);
}

/**
 * @brief Create a matrix representation of the controlled phase-shift gate data
in row-major format.
 *
 * @see `getPhaseShift<T,U>(U angle)`.
 */
template <class T, class U = T>
static const std::vector<std::complex<T>> getControlledPhaseShift(U angle) {
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
static const std::vector<std::complex<T>>
getControlledPhaseShift(const std::vector<U> &params) {
    return getControlledPhaseShift<T>(params.front());
}

} // namespace Gates
} // namespace Pennylane
