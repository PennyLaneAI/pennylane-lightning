#include "tensor.hpp"
#include <iostream>
#include <sstream> 
#include <vector>

// Helper to check CUDA calls for errors
void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(result)));
    }
}

DeviceTensor::DeviceTensor(size_t size) : m_size(size) {
    if (m_size > 0) {
        check_cuda(cudaMalloc(&p_data, m_size * sizeof(float)));
        std::cout << "Allocated a DeviceTensor on GPU with " << m_size << " elements." << std::endl;
    }
}

DeviceTensor::~DeviceTensor() {
    if (p_data) {
        cudaFree(p_data);
        std::cout << "Deallocated a DeviceTensor." << std::endl;
        p_data = nullptr;
    }
}

// Move constructor
DeviceTensor::DeviceTensor(DeviceTensor&& other) noexcept
    : p_data(other.p_data), m_size(other.m_size) {
    // Leave the moved-from object in a valid but empty state
    other.p_data = nullptr;
    other.m_size = 0;
}

// Move assignment
DeviceTensor& DeviceTensor::operator=(DeviceTensor&& other) noexcept {
    if (this != &other) {
        // Free existing resource
        if (p_data) {
            cudaFree(p_data);
        }
        // Pilfer other's resource
        p_data = other.p_data;
        m_size = other.m_size;

        // Reset other
        other.p_data = nullptr;
        other.m_size = 0;
    }
    return *this;
}

// Python __repr__ method for string representation, only give size
std::string DeviceTensor::repr() const {
    std::ostringstream oss;
    oss << "DeviceTensor([" << m_size << "])";
    return oss.str();
}

// Print the raw data stored within GPU memory (must copy to CPU first)
void DeviceTensor::print_data() const {
    if (p_data) {
        // Copy to host (expensive)
        std::vector<float> host_data(m_size);
        check_cuda(cudaMemcpy(host_data.data(), p_data, m_size * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "DeviceTensor data: ";
        for (const auto& val : host_data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "DeviceTensor is empty." << std::endl;
    }
}