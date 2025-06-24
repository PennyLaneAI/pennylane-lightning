#pragma once
#include <complex>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "Util.hpp"

namespace nb = nanobind;

namespace Pennylane::NanoBindings::Utils {

/**
 * @brief Create an ndarray from a vector of data with proper ownership transfer
 *
 * @tparam VectorT Data type of the vector elements
 * @param data Vector containing the data to transfer
 * @param shape Vector containing the shape of the resulting array
 * @return nb::ndarray<VectorT, nb::numpy, nb::c_contig> Array with copied data
 * in numpy format
 */
template <typename VectorT>
nb::ndarray<VectorT, nb::numpy, nb::c_contig>
createNumpyArrayFromVector(const std::vector<VectorT> &data,
                           const std::vector<size_t> &shape) {
    // Calculate total size from shape
    size_t total_size = 1;
    for (const auto &dim : shape) {
        total_size *= dim;
    }

    // Verify data size matches shape
    if (data.size() != total_size) {
        throw std::runtime_error(
            "Data size does not match the specified shape");
    }

    std::vector<VectorT> *new_data = new std::vector<VectorT>(std::move(data));

    // Create a capsule to manage memory
    auto capsule = nb::capsule(new_data, [](void *p) noexcept {
        delete static_cast<std::vector<VectorT> *>(p);
    });

    // Create and return the ndarray with numpy format
    return nb::ndarray<VectorT, nb::numpy, nb::c_contig>(
        new_data->data(), shape.size(), shape.data(), capsule);
}

/**
 * @brief Create a 1D ndarray from a vector of data
 *
 * @tparam VectorT Data type of the vector elements
 * @param data Vector containing the data to transfer
 * @return nb::ndarray<VectorT, nb::numpy, nb::c_contig> 1D array with copied
 * data
 */
template <typename VectorT>
nb::ndarray<VectorT, nb::numpy, nb::c_contig>
createNumpyArrayFromVector(const std::vector<VectorT> &data) {
    return createNumpyArrayFromVector<VectorT>(data, {data.size()});
}

/**
 * @brief Create a 2D ndarray from a vector of data
 *
 * @tparam VectorT Data type of the vector elements
 * @param data Vector containing the data to transfer
 * @param rows Number of rows in the resulting 2D array
 * @param cols Number of columns in the resulting 2D array
 * @return nb::ndarray<VectorT, nb::numpy, nb::c_contig> 2D array with copied
 * data
 */
template <typename VectorT>
nb::ndarray<VectorT, nb::numpy, nb::c_contig>
createNumpyArrayFromVector(const std::vector<VectorT> &data, std::size_t rows,
                           std::size_t cols) {
    return createNumpyArrayFromVector<VectorT>(data, {rows, cols});
}

/**
 * @brief Convert complex matrices from nanobind ndarray to std::vector
 *
 * @tparam ComplexT Complex type to convert to
 * @tparam ParamT Precision type of the complex numbers
 * @param matrices Vector of ndarrays containing matrices
 * @return std::vector<std::vector<ComplexT>> Converted matrices
 */
template <typename ComplexT, typename ParamT>
std::vector<std::vector<ComplexT>> convertMatrices(
    const std::vector<nb::ndarray<const std::complex<ParamT>, nb::c_contig>>
        &matrices) {
    std::vector<std::vector<ComplexT>> conv_matrices(matrices.size());

    for (std::size_t i = 0; i < matrices.size(); i++) {
        if (matrices[i].size() > 0) {
            const auto *m_ptr = matrices[i].data();
            const auto m_size = matrices[i].size();
            conv_matrices[i] = std::vector<ComplexT>(m_ptr, m_ptr + m_size);
        }
    }

    return conv_matrices;
}

/**
 * @brief Generate string representation of operations data
 *
 * @tparam OpsDataT Type of operations data
 * @param ops Operations data object
 * @param include_controlled Whether to include controlled wires and values
 * @return std::string String representation
 */
template <typename OpsDataT>
std::string opsDataToString(const OpsDataT &ops,
                            bool include_controlled = true) {
    using namespace Pennylane::Util;
    std::ostringstream ops_stream;
    for (std::size_t op = 0; op < ops.getSize(); op++) {
        ops_stream << "{'name': " << ops.getOpsName()[op];
        ops_stream << ", 'params': " << ops.getOpsParams()[op];
        ops_stream << ", 'inv': " << ops.getOpsInverses()[op];

        if (include_controlled) {
            ops_stream << ", 'controlled_wires': "
                       << ops.getOpsControlledWires()[op];
            ops_stream << ", 'controlled_values': "
                       << ops.getOpsControlledValues()[op];
        }

        ops_stream << ", 'wires': " << ops.getOpsWires()[op];
        ops_stream << "}";
        if (op < ops.getSize() - 1) {
            ops_stream << ",";
        }
    }
    return "Operations: [" + ops_stream.str() + "]";
}

} // namespace Pennylane::NanoBindings::Utils
