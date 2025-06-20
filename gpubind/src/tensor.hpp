#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <nanobind/ndarray.h> // For the to_cpu method
#include <nanobind/stl/string.h>
#include <numeric>
#include <stdexcept>

namespace nb = nanobind;

// Wrapper class for CUDA device memory
class DeviceTensor {
  public:
    // Constructor: allocates memory on the GPU
    DeviceTensor(size_t n_size);

    // Destructor: frees memory on the GPU
    ~DeviceTensor();

    // Deleted copy constructor and assignment operator to prevent
    // double-freeing memory
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    DeviceTensor(DeviceTensor &&other) noexcept;
    DeviceTensor &operator=(DeviceTensor &&other) noexcept;

    float *data() const { return p_data; }
    size_t size() const { return m_size; }
    void print_data() const;

    // Python __repr__ method for string representation
    std::string repr() const;

  private:
    float *p_data = nullptr;
    size_t m_size;
};