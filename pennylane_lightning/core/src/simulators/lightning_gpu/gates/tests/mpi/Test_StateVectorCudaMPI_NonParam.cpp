#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>
#include <mpi.h>

#include "cuGateCache.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

#include "StateVectorCudaMPI.hpp"
#include "StateVectorCudaManaged.hpp"
//#include "StateVectorLQubitRaw.hpp"

#include "MPIManager.hpp"

//#include "../TestHelpersLGPU.hpp"
#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::LightningGPU;
//using namespace CUDA;

#define num_qubits 8
#define lsb_1qbit                                                              \
    { 0 }
#define msb_1qbit                                                              \
    { num_qubits - 1 }
#define lsb_2qbit                                                              \
    { 0, 1 }
#define msb_2qubit                                                             \
    { num_qubits - 2, num_qubits - 1 }
#define mlsb_2qubit                                                            \
    { 0, num_qubits - 1 }
#define lsb_3qbit                                                              \
    { 0, 1, 2 }
#define msb_3qubit                                                             \
    { num_qubits - 3, num_qubits - 2, num_qubits - 1 }
#define mlsb_3qubit                                                            \
    { 0, num_qubits - 2, num_qubits - 1 }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetStateVector",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    std::vector<cp_t> init_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits));
    std::vector<cp_t> local_state(subSvLength);

    using index_type =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    std::vector<index_type> indices(Pennylane::Util::exp2(num_qubits));

    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        auto st = Pennylane::Util::createRandomStateVectorData<PrecisionT>(re, num_qubits);
        init_state.clear();
        init_state = std::vector<cp_t>(st.begin(), st.end(), init_state.get_allocator());
        expected_state = init_state;
        for (size_t i = 0; i < Pennylane::Util::exp2(num_qubits - 1); i++) {
            std::swap(expected_state[i * 2], expected_state[i * 2 + 1]);
            indices[i * 2] = i * 2 + 1;
            indices[i * 2 + 1] = i * 2;
        }
    }
    mpi_manager.Barrier();

    auto expected_local_state = mpi_manager.scatter<cp_t>(expected_state, 0);
    mpi_manager.Bcast<index_type>(indices, 0);
    mpi_manager.Bcast<cp_t>(init_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    //`values[i]` on the host will be copy the `indices[i]`th element of the
    // state vector on the device.
    SECTION("Set state vector with values and their corresponding indices on "
            "the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nGlobalIndexBits, nLocalIndexBits);
        // The setStates will shuffle the state vector values on the device with
        // the following indices and values setting on host. For example, the
        // values[i] is used to set the indices[i] th element of state vector on
        // the device. For example, values[2] (init_state[5]) will be copied to
        // indices[2]th or (4th) element of the state vector.

        sv.template setStateVector<index_type>(
            init_state.size(), init_state.data(), indices.data(), false);

        mpi_manager.Barrier();
        sv.CopyGpuDataToHost(local_state.data(),
                             static_cast<std::size_t>(subSvLength));
        mpi_manager.Barrier();

        CHECK(expected_local_state == Pennylane::Util::approx(local_state));
    }
}
/*
TEMPLATE_TEST_CASE("StateVectorCudaMPI::SetIthStates",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    using PrecisionT = TestType;
    using cp_t = std::complex<PrecisionT>;
    MPIManager mpi_manager(MPI_COMM_WORLD);

    size_t mpi_buffersize = 1;

    size_t nGlobalIndexBits =
        std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;
    size_t nLocalIndexBits = num_qubits - nGlobalIndexBits;
    size_t subSvLength = 1 << nLocalIndexBits;
    mpi_manager.Barrier();

    int index;
    if (mpi_manager.getRank() == 0) {
        std::mt19937 re{1337};
        std::uniform_int_distribution<> distr(
            0, Pennylane::Util::exp2(num_qubits) - 1);
        index = distr(re);
    }
    mpi_manager.Bcast(index, 0);

    std::vector<cp_t> expected_state(Pennylane::Util::exp2(num_qubits), {0, 0});
    if (mpi_manager.getRank() == 0) {
        expected_state[index] = {1.0, 0};
    }

    auto expected_local_state = mpi_manager.scatter(expected_state, 0);
    mpi_manager.Barrier();

    int nDevices = 0; // Number of GPU devices per node
    cudaGetDeviceCount(&nDevices);
    int deviceId = mpi_manager.getRank() % nDevices;
    cudaSetDevice(deviceId);
    DevTag<int> dt_local(deviceId, 0);

    SECTION(
        "Set Ith element of the state state on device with data on the host") {
        StateVectorCudaMPI<PrecisionT> sv(mpi_manager, dt_local, mpi_buffersize,
                                          nGlobalIndexBits, nLocalIndexBits);
        std::complex<PrecisionT> values = {1.0, 0};
        sv.setBasisState(values, index, false);

        std::vector<cp_t> h_sv0(subSvLength, {0.0, 0.0});
        sv.CopyGpuDataToHost(h_sv0.data(),
                             static_cast<std::size_t>(subSvLength));

        CHECK(expected_local_state == Pennylane::Util::approx(h_sv0));
    }
}


#define PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, NUM_QUBITS, GATE_METHOD,    \
                                         GATE_NAME, WIRE)                      \
    {                                                                          \
        using cp_t = std::complex<TestType>;                                   \
        using PrecisionT = TestType;                                           \
        MPIManager mpi_manager(MPI_COMM_WORLD);                                \
        size_t mpi_buffersize = 1;                                             \
        size_t nGlobalIndexBits =                                              \
            std::bit_width(static_cast<size_t>(mpi_manager.getSize())) - 1;    \
        size_t nLocalIndexBits = (NUM_QUBITS)-nGlobalIndexBits;                \
        size_t subSvLength = 1 << nLocalIndexBits;                             \
        size_t svLength = 1 << (NUM_QUBITS);                                   \
        mpi_manager.Barrier();                                                 \
        std::vector<cp_t> init_sv(svLength);                                   \
        std::vector<cp_t> expected_sv(svLength);                               \
        if (mpi_manager.getRank() == 0) {                                      \
            std::mt19937 re{1337};                                             \
            auto random_sv =                                                   \
            Pennylane::Util::createRandomStateVectorData<PrecisionT>(re, (NUM_QUBITS));  \
            init_sv = random_sv;                                               \
        }                                                                      \
        auto local_state = mpi_manager.scatter(init_sv, 0);                    \
        mpi_manager.Barrier();                                                 \
        int nDevices = 0;                                                      \
        cudaGetDeviceCount(&nDevices);                                         \
        int deviceId = mpi_manager.getRank() % nDevices;                       \
        cudaSetDevice(deviceId);                                               \
        DevTag<int> dt_local(deviceId, 0);                                     \
        mpi_manager.Barrier();                                                 \
        SECTION("Apply directly") {                                            \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.GATE_METHOD(WIRE, false);                                   \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                                                                               \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.GATE_METHOD(WIRE, false);                    \
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),        \
                                                    svLength);                 \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::Util::approx(expected_local_sv));\
            }                                                                  \
        }                                                                      \
        SECTION("Apply using dispatcher") {                                    \
            SECTION("Operation on target wire") {                              \
                StateVectorCudaMPI<TestType> sv(                               \
                    mpi_manager, dt_local, mpi_buffersize, nGlobalIndexBits,   \
                    nLocalIndexBits);                                          \
                sv.CopyHostDataToGpu(local_state, false);                      \
                sv.applyOperation(GATE_NAME, WIRE, false);                     \
                sv.CopyGpuDataToHost(local_state.data(),                       \
                                     static_cast<std::size_t>(subSvLength));   \
                SVDataGPU<TestType> svdat{(NUM_QUBITS), init_sv};              \
                if (mpi_manager.getRank() == 0) {                              \
                    svdat.cuda_sv.applyOperation(GATE_NAME, WIRE, false);      \
                    svdat.cuda_sv.CopyGpuDataToHost(expected_sv.data(),        \
                                                    svLength);                 \
                }                                                              \
                auto expected_local_sv = mpi_manager.scatter(expected_sv, 0);  \
                CHECK(local_state == Pennylane::Util::approx(expected_local_sv));\
            }                                                                  \
        }                                                                      \
    }

TEMPLATE_TEST_CASE("StateVectorCudaMPI::PauliX",
                   "[StateVectorCudaMPI_Nonparam]", float, double) {
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", lsb_1qbit);
    PLGPU_MPI_TEST_GATE_OPS_NONPARAM(TestType, num_qubits, applyPauliX,
                                     "PauliX", {num_qubits - 1});
}
*/