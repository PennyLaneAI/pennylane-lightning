// Copyright 2021 Xanadu Quantum Technologies Inc.
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

#include <cuComplex.h>
#include <cuda.h>

#include "StateVector.hpp"

namespace {
cuDoubleComplex getCudaType(const double &t) {
    static_cast<void>(t);
    return {};
}
cuComplex getCudaType(const float &t) {
    static_cast<void>(t);
    return {};
}
} // namespace

namespace Pennylane {

/**
 * @brief GPU managed memory version of StateVector class.
 *
 * @tparam fp_t
 */
template <class fp_t = double> class StateVectorGPU : public StateVector<fp_t> {
  private:
    using CFP_t = decltype(getCudaType(fp_t{}));
    CFP_t *arr_;

    inline void CopyHostDataToGpu(const StateVector<fp_t> &sv) {
        cudaMemcpy(arr_, sv.getData(), sizeof(CFP_t) * sv.getLength(),
                   cudaMemcpyHostToDevice);
    }
    inline void CopyGpuDataToHost(StateVector<fp_t> &sv) const {
        cudaMemcpy(sv.getData(), arr_, sizeof(CFP_t) * this->getLength(),
                   cudaMemcpyDeviceToHost);
    }
    inline void CopyGpuDataToGpu(StateVectorGPU<fp_t> &sv) {
        cudaMemcpy(sv.getData(), arr_, sizeof(CFP_t) * this->getLength(),
                   cudaMemcpyDeviceToDevice);
    }
    inline void AsyncCopyHostDataToGpu(const StateVector<fp_t> &sv,
                                       cudaStream_t stream = 0) {
        cudaMemcpyAsync(arr_, sv.getData(), sizeof(CFP_t) * sv.getLength(),
                        cudaMemcpyHostToDevice, stream);
    }
    inline void AsyncCopyGpuDataToHost(StateVector<fp_t> &sv,
                                       cudaStream_t stream = 0) {
        cudaMemcpyAsync(sv.getData(), arr_, sizeof(CFP_t) * this->getLength(),
                        cudaMemcpyDeviceToHost, stream);
    }

  public:
    StateVectorGPU() : StateVector<fp_t>(), arr_{nullptr} {}
    StateVectorGPU(CFP_t *arr, size_t length)
        : StateVector<fp_t>(nullptr, length) {
        cudaMalloc(reinterpret_cast<void **>(&arr_), sizeof(CFP_t));
    }
    ~StateVectorGPU() { cudaFree(arr_); }

    template <class CPUData>
    StateVectorGPU &operator=(const StateVector<CPUData> &other) {
        static_assert(sizeof(CPUData) == sizeof(fp_t),
                      "Size of CPU and GPU data types do not match.");
        CopyHostDataToGpu(other);
        return *this;
    }

    auto getDataVector() -> std::vector<std::complex<fp_t>> {
        std::vector<std::complex<fp_t>> host_data_buffer(this->getLength());
        auto ptr = reinterpret_cast<CFP_t *>(host_data_buffer.data());
        cudaMemcpy(ptr, arr_, sizeof(CFP_t) * this->getLength(),
                   cudaMemcpyDeviceToHost);
        return host_data_buffer;
    }
};

} // namespace Pennylane