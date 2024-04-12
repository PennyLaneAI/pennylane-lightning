/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cutensornet.h>

#include "MPSManagedNew.hpp"

// Sphinx: #10
int main() {
    using ComplexT = MPSHelper<double>::ComplexT;

    const size_t cuTensornetVersion = cutensornetGetVersion();
    printf("cuTensorNet-vers:%ld\n", cuTensornetVersion);

    cudaDeviceProp prop;
    int deviceId{-1};
    HANDLE_CUDA_ERROR(cudaGetDevice(&deviceId));
    HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceId));

    // Sphinx: #11
    /***********************************
     * Step 1: basic MPS setup
     ************************************/

    // setup the simulation setting for the MPS
    // typedef std::complex<double> complexType;
    int32_t numSites = 16;
    int64_t physExtent = 2;
    int64_t maxVirtualExtent = 12;
    const std::vector<int64_t> initialVirtualExtents(
        numSites - 1, 1); // starting MPS with shared extent of 1;

    // initialize an MPSHelper to dynamically update tensor metadats
    MPSHelper<double> mpsHelper(numSites, physExtent, maxVirtualExtent,
                                initialVirtualExtents);

    mpsHelper.initTensorStates();

    // initialize 4 random gate tensors on host and copy them to device
    const int32_t numRandomGates = 4;
    const int64_t numGateElements =
        physExtent * physExtent * physExtent * physExtent; // shape (2, 2, 2, 2)
    size_t gateSize = sizeof(ComplexT) * numGateElements;
    ComplexT *gates_h[numRandomGates];
    void *gates_d[numRandomGates];

    for (int i = 0; i < numRandomGates; i++) {
        gates_h[i] = (ComplexT *)malloc(gateSize);
        HANDLE_CUDA_ERROR(cudaMalloc((void **)&gates_d[i], gateSize));
        for (int j = 0; j < numGateElements; j++) {
            gates_h[i][j] = ComplexT(((float)rand()) / RAND_MAX,
                                     ((float)rand()) / RAND_MAX);
        }
        HANDLE_CUDA_ERROR(cudaMemcpy(gates_d[i], gates_h[i], gateSize,
                                     cudaMemcpyHostToDevice));
    }

    // Sphinx: #15
    /***********************************
     * Step 5: execution
     ************************************/
    uint32_t numLayers = 10; // 10 layers of gate
    for (uint32_t i = 0; i < numLayers; i++) {
        uint32_t start_site = i % 2;
        std::cout << "Cycle " << i << ":" << std::endl;
        // bool verbose = (i == numLayers - 1);
        for (int32_t j = start_site; j < numSites - 1; j = j + 2) {
            uint32_t gateIdx =
                rand() % numRandomGates; // pick a random gate tensor
            std::cout << "apply gate " << gateIdx << " on " << j << " and "
                      << j + 1 << std::endl;

            void *dataG = gates_d[gateIdx];
            mpsHelper.applyGate(j, j + 1, dataG);
        }
    }

    std::cout << "Free all resources" << std::endl;

    for (int i = 0; i < numRandomGates; i++) {
        free(gates_h[i]);
        HANDLE_CUDA_ERROR(cudaFree(gates_d[i]));
    }

    return 0;
}
