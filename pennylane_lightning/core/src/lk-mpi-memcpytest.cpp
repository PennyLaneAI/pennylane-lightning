#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#include <mpi.h>

#include "Constant.hpp"
#include "ConstantUtil.hpp"
#include "Error.hpp"
#include "MeasurementsKokkos.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"
#include <iostream>
#include <rocprim/intrinsics.hpp>
#include <roctracer/roctx.h>

namespace {
using namespace Pennylane;
using namespace Pennylane::Gates;
using namespace Pennylane::Gates::Constant;
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::Util;
using t_scale = std::milli;
using KokkosVector = Kokkos::View<Kokkos::complex<double> *>;
}


std::size_t  prep_input_1q(int argc, char *argv[]) {
    if (argc <= 1) {
        std::cout << "Please ensure you specify the following arguments: "
                     "total_qubits"
                  << std::endl;
        std::exit(-1);
    }
    std::string arg_qubits = argv[1];
    std::size_t qubits = std::stoi(arg_qubits);

    return qubits;
}




int main(int argc, char *argv[]) {


    std::size_t nq = prep_input_1q(argc, argv);

    // Create PennyLane Lightning statevector
    StateVectorKokkos<double> sv(nq);
    std::unique_ptr<KokkosVector> data_ = std::make_unique<KokkosVector>("data_", 1U << (nq));

    std::unique_ptr<KokkosVector> sendbuf_ = std::make_unique<KokkosVector>("sendbuf_", 1U << (nq - 1));
    std::unique_ptr<KokkosVector> recvbuf_ = std::make_unique<KokkosVector>("recvbuf_", 1U << (nq - 1));

    uint32_t repeats = 5;

    // Fill vector with {0, 1, 2, ... nq-1}
    std::vector<uint32_t> index_not_swapped;
    index_not_swapped.resize(nq - 1);
    std::iota(index_not_swapped.begin(), index_not_swapped.end(), 0);

    const uint32_t not_swapping_local_wire_size = index_not_swapped.size(); 
    //auto index_not_swapped_view = vector2view(index_not_swapped);

    using UnmanagedView = Kokkos::View<const uint32_t *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;


    Kokkos::View<uint32_t *> index_not_swapped_view("index_not_swapped", not_swapping_local_wire_size);
    Kokkos::deep_copy(index_not_swapped_view, UnmanagedView(index_not_swapped.data(), not_swapping_local_wire_size));
    

    uint32_t swap_wire_mask = 1U << (nq - 1);

    auto sendbuf_view = (*sendbuf_);
    auto recvbuf_view = (*recvbuf_);
    auto sv_view = (*data_);
    // Warmup
    roctxMark("ROCTX-MARK: before warmup");
        Kokkos::parallel_for("copy_sendbuf",
            1U << (nq - 1),
            KOKKOS_LAMBDA(uint32_t buffer_index) {
                //uint32_t SV_index = swap_wire_mask;
                //for (int i = 0;
                //        i < not_swapping_local_wire_size;
                //        i++) {
                //    SV_index |=
                //        (((buffer_index >> i) & 1)
                //            << index_not_swapped_view(i));
                //}
               //SV_index = buffer_index ^ (1U << 3) ^ (1U);
               uint32_t SV_index = buffer_index;
               Kokkos::complex<double> val = sv_view(SV_index);
               rocprim::wave_barrier();
               sendbuf_view(buffer_index) = val;
            });
        Kokkos::fence();

    roctxMark("ROCTX-MARK: after warmup");

    const auto t_start = std::chrono::high_resolution_clock::now();   

    for (uint32_t i = 0; i < repeats; i++) {   

    roctxMark("ROCTX-MARK: Starting iteration");
        Kokkos::parallel_for("copy_sendbuf",
            1U << (nq - 1),
            KOKKOS_LAMBDA(uint32_t buffer_index) {
                //uint32_t SV_index = swap_wire_mask;
                //for (int i = 0;
                //        i < not_swapping_local_wire_size;
                //        i++) {
                //    SV_index |=
                //        (((buffer_index >> i) & 1)
                //            << index_not_swapped_view(i));
                //}
               //SV_index = buffer_index ^ (1U << 3) ^ (1U);
               uint32_t SV_index = buffer_index;
               Kokkos::complex<double> val = sv_view(SV_index);
               //rocprim::wave_barrier();
               sendbuf_view(buffer_index) = val;
            });
        Kokkos::fence();

    roctxMark("ROCTX-MARK: ending iteration");
    }
    const auto t_end = std::chrono::high_resolution_clock::now();   
    const double t_duration = std::chrono::duration<double, t_scale>(t_end - t_start).count();  
    double average_time = t_duration / (repeats); 
    double data_copied_GB = exp2(nq - 1) * 128 / 8 / 1024 / 1024 / 1024;
    std::cout << "Average time for copying "  << average_time << " ms" << std::endl;  
    std::cout << "Data copied = " << data_copied_GB  << " GB" << std::endl;  
    std::cout << "Effective copy speed = " << data_copied_GB/average_time*1000.0 << " GB/s " << std::endl;  


    return 0;
}
