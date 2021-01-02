#pragma once

#include <array>
#include <complex>
#include "unsupported/Eigen/CXX11/Tensor"

template<int X>
using State_Xq = Eigen::Tensor<std::complex<double>, X>;

template<int X>
using Gate_Xq = Eigen::Tensor<std::complex<double>, 2 * X>;

using Pairs = Eigen::IndexPair<int>;
template<int X>
using Pairs_Xq = std::array<Eigen::IndexPair<int>, X>;

// Creating aliases based on the function signatures of each operation

template<int X>
using pfunc_Xq = Gate_Xq<X>(*)();

template<int X>
using pfunc_Xq_one_param = Gate_Xq<X>(*)(const double&);

template<int X>
using pfunc_Xq_three_params = Gate_Xq<X>(*)(const double&, const double&, const double&);