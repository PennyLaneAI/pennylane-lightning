#include <numeric>                        // Standard library import for std::accumulate
#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings
#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "pybind11/stl.h"
#include "xtensor/xsort.hpp"
#define XTENSOR_USE_XSIMD

#include "pybind11/eigen.h"
#include <iostream>
#include <Eigen/Dense>
//#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <unsupported/Eigen/CXX11/Tensor>
//#include <bitset>

using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::Tensor;



VectorXcd test(VectorXcd state)
{
  MatrixXd X(2,2);
  X(0,0) = 0;
  X(1,0) = 1;
  X(0,1) = 1;
  X(1,1) = 0;

  return X * state;
}


//template<class T>
//py::array_t<T> eigenTensor(py::array_t<T> inArray)
//Tensor<std::complex<double>, 2> test(Tensor<std::complex<double>, 2> state)
//{
//  return state;
//}


//VectorXcd test(VectorXcd state)
//{
//  MatrixXd X(2,2);
//  X(0,0) = 0;
//  X(1,0) = 1;
//  X(0,1) = 1;
//  X(1,1) = 0;
//
//  std::string binary = std::bitset<8>(128).to_string(); //to binary
//  std::cout<<binary<<"\n";
//
//  unsigned long decimal = std::bitset<8>(binary).to_ulong();
//  std::cout<<decimal<<"\n";
//
//  return X * state;
//}

//VectorXcd test(VectorXcd state)
//{
//  MatrixXd X(2,2);
//  X(0,0) = 0;
//  X(1,0) = 1;
//  X(0,1) = 1;
//  X(1,1) = 0;
//  return X * state;
//}


//Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> test(
//    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> op,
//    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> state,
//    std::vector<unsigned long int>& op_wires
//)
//{
//  MatrixXd m(2,2);
//  m(0,0) = 3;
//  m(1,0) = 2.5;
//  m(0,1) = -1;
//  m(1,1) = m(1,0) + m(0,1);
//  return state;
//}

xt::pyarray<std::complex<double>> mvp(xt::pyarray<std::complex<double>>& op, xt::pyarray<std::complex<double>>& state,
std::vector<unsigned long int>& op_wires)
{
    unsigned long int length = op.shape().size() * 0.5;
    std::vector<unsigned long int> axis(length);

    unsigned long int n_wires = state.shape().size();
    std::vector<unsigned long int> ordering = op_wires;

    for (int i=0; i<n_wires; i++){
        if (i < length) {axis[i] = i + length;}
        if (std::find(op_wires.begin(), op_wires.end(), i) == op_wires.end())
        {
            ordering.insert(ordering.end(), i);
        }
    }

    auto tdot = xt::linalg::tensordot(op, state, axis, op_wires);

    std::vector<unsigned long int> ordering_shape = {n_wires};
    auto ordering_xt = xt::adapt(ordering, ordering_shape);
    auto reordering = xt::argsort(ordering_xt);
    auto tdot_reordered = xt::transpose(tdot, reordering);

    return tdot_reordered;
}

PYBIND11_MODULE(lightning_qubit_ops, m)
{
    xt::import_numpy();
    m.doc() = "Lightning qubit operations using XTensor";
    m.def("mvp", mvp, "Matrix vector product");
    m.def("test", test, "test");
}
