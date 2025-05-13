#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <string_view>
#include <vector>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <iostream>
// #include <rocprim/intrinsics.hpp>
// #include <roctracer/roctx.h>

namespace {
using t_scale = std::milli;
using KokkosVector = Kokkos::View<Kokkos::complex<double> *>;
} // namespace

std::size_t prep_input_1q(int argc, char *argv[]) {
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

    Kokkos::initialize();
    {
        KokkosVector data_("data_", 1U << (nq));
        KokkosVector sendbuf_("sendbuf_", 1U << (nq - 1));
        KokkosVector recvbuf_("recvbuf_", 1U << (nq - 1));
        std::size_t repeats = 5;
        std::vector<std::size_t> index_not_swapped;
        index_not_swapped.resize(nq - 1);

        // right most index
        // std::iota(index_not_swapped.begin(), index_not_swapped.end(), 0);
        // std::size_t swap_wire_mask = 1U << (nq - 1);

        // left most index
        std::iota(index_not_swapped.rbegin(), index_not_swapped.rend(), 1);
        std::size_t swap_wire_mask = 1U;

        std::cout << "swap_wire_mask = " << swap_wire_mask << std::endl;
        std::cout << "index_not_swapped = ";
        for (auto &i : index_not_swapped) {
            std::cout << i << " ";
        }

        const std::size_t not_swapping_local_wire_size =
            index_not_swapped.size();
        using UnmanagedView =
            Kokkos::View<const std::size_t *, Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

        Kokkos::View<std::size_t *> index_not_swapped_view(
            "index_not_swapped", not_swapping_local_wire_size);
        Kokkos::deep_copy(index_not_swapped_view,
                          UnmanagedView(index_not_swapped.data(),
                                        not_swapping_local_wire_size));

        std::size_t buffer_size = 1U << (nq - 1);
        std::cout << "Buffer size = " << buffer_size << std::endl;

        // Warmup
        // roctxMark("ROCTX-MARK: before warmup");
        Kokkos::parallel_for(
            "copy_sendbuf", buffer_size,
            KOKKOS_LAMBDA(std::size_t buffer_index) {
                sendbuf_(buffer_index) = data_(buffer_index);
            });
        Kokkos::fence();
        // roctxMark("ROCTX-MARK: after warmup");

        auto t_start = std::chrono::high_resolution_clock::now();

        for (std::size_t i = 0; i < repeats; i++) {

            // roctxMark("ROCTX-MARK: Starting iteration");

            Kokkos::parallel_for(
                "copy_sendbuf", buffer_size,
                KOKKOS_LAMBDA(std::size_t buffer_index) {
                    std::size_t SV_index = swap_wire_mask;
                    for (int i = 0; i < not_swapping_local_wire_size; i++) {
                        SV_index |= (((buffer_index >> i) & 1)
                                     << index_not_swapped_view(i));
                    }
                    sendbuf_(buffer_index) = data_(SV_index);
                });
            Kokkos::fence();
            /*  Kokkos::parallel_for("copy_recvbuf",
                 buffer_size,
                 KOKKOS_LAMBDA(std::size_t buffer_index) {
                 std::size_t SV_index = swap_wire_mask;
                 for (int i = 0;
                         i < not_swapping_local_wire_size;
                         i++) {
                     SV_index |=
                         (((buffer_index >> i) & 1)
                             << index_not_swapped_view(i));
                 }
                 data_(SV_index) = recvbuf_(buffer_index);
                 });
                 Kokkos::fence(); */
        }
        Kokkos::fence();
        auto t_end = std::chrono::high_resolution_clock::now();

        double t_duration =
            std::chrono::duration<double, t_scale>(t_end - t_start).count();
        double average_time = t_duration / (repeats);
        double data_copied_GB = static_cast<double>(buffer_size) * 128.0 / 8.0 /
                                1024.0 / 1024.0 / 1024.0;
        std::cout << "Average time for copying to sendbuf" << average_time
                  << " ms" << std::endl;
        std::cout << "Data copied = " << data_copied_GB << " GB" << std::endl;
        std::cout << "Effective copy speed = "
                  << data_copied_GB / average_time * 1000.0 << " GB/s "
                  << std::endl;
    }
    if (!Kokkos::is_finalized()) {
        Kokkos::finalize();
    }
    return 0;
}
