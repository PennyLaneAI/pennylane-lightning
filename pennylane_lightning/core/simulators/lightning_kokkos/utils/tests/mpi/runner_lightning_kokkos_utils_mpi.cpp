#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize();
    int result = Catch::Session().run(argc, argv);
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
