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

template <class T> static constexpr std::vector<std::complex<T>> getPauliX() {
    return {ZERO<T>(), ONE<T>(), ONE<T>(), ZERO<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getPauliY() {
    return {ZERO<T>(), -IMAG<T>(), IMAG<T>(), ZERO<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getPauliZ() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), -ONE<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getHadamard() {
    return {INVSQRT2<T>(), INVSQRT2<T>(), INVSQRT2<T>(), -INVSQRT2<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getS() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), IMAG<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getT() {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), IMAG<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getCNOT() {
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getSWAP() {
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(),  ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(), ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>()};
}

template <class T> static constexpr std::vector<std::complex<T>> getCZ() {
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),  ZERO<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), -ONE<T>()};
}

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

template <class T, class U = T>
static const std::vector<std::complex<T>> getPhaseShift(U angle) {
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), std::exp(IMAG<T>() * angle)};
}

template <class T, class U = T>
static const std::vector<std::complex<T>>
getPhaseShift(const std::vector<U> &params) {
    return getPhaseShift<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRX(U angle) {
    const std::complex<T> c(std::cos(angle / 2), 0);
    const std::complex<T> js(0, -std::sin(angle / 2));
    return {c, js, js, c};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRX(const std::vector<U> &params) {
    return getRX<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRY(U angle) {
    const std::complex<T> c(std::cos(angle / 2), 0);
    const std::complex<T> s(std::sin(angle / 2), 0);
    return {c, -s, s, c};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRY(const std::vector<U> &params) {
    return getRY<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRZ(U angle) {
    return {std::exp(-IMAG<T>() * (angle / 2)), ZERO<T>(), ZERO<T>(),
            std::exp(IMAG<T>() * (angle / 2))};
}

template <class T, class U = T>
static const std::vector<T> getRZ(const std::vector<U> &params) {
    return getRZ<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRot(U phi, U theta, U omega) {
    const std::complex<T> c(std::cos(theta / 2), 0), s(std::sin(theta / 2), 0);
    const U p{phi + omega}, m{phi - omega};
    return {std::exp(static_cast<T>(p / 2) * (-IMAG<T>())) * c,
            -std::exp(static_cast<T>(m / 2) * IMAG<T>()) * s,
            std::exp(static_cast<T>(m / 2) * (-IMAG<T>())) * s,
            std::exp(static_cast<T>(p / 2) * IMAG<T>()) * c};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getRot(const std::vector<U> &params) {
    return getRot<T>(params[0], params[1], params[2]);
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRX(U angle) {
    const std::complex<T> c(std::cos(angle / 2), 0),
        js(0, std::sin(-angle / 2));
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), c,         js,
            ZERO<T>(), ZERO<T>(), js,        c};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRX(const std::vector<U> &params) {
    return getCRX<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRY(U angle) {
    const std::complex<T> c(std::cos(angle / 2), 0), s(std::sin(angle / 2), 0);
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), c,         -s,
            ZERO<T>(), ZERO<T>(), s,         c};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRY(const std::vector<U> &params) {
    return getCRY<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRZ(U angle) {
    const std::complex<T> first = std::exp(-IMAG<T>() * (angle / 2));
    const std::complex<T> second = std::exp(IMAG<T>() * (angle / 2));
    return {ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            ONE<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(),
            first,    ZERO<T>(), ZERO<T>(), ZERO<T>(), second};
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRZ(const std::vector<U> &params) {
    return getCRZ<T>(params.front());
}

template <class T, class U = T>
static const std::vector<std::complex<T>> getCRot(U phi, U theta, U omega) {
    const std::vector<std::complex<T>> rot = getRot<T>(phi, theta, omega);
    return {ONE<T>(),  ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), ONE<T>(),
            ZERO<T>(), ZERO<T>(), ZERO<T>(), ZERO<T>(), rot[0],    rot[1],
            ZERO<T>(), ZERO<T>(), rot[2],    rot[3]};
}

template <class T, class U = T>
static const std::vector<std::complex<T>>
getCRot(const std::vector<U> &params) {
    return getCRot<T>(params[0], params[1], params[2]);
}

} // namespace Gates
} // namespace Pennylane