BUILDDIR=build_lightning_kokkos
cmake -B${BUILDDIR} -S . -DPL_BACKEND=lightning_kokkos -DKokkos_ENABLE_CUDA=ON -DENABLE_PYTHON=OFF
cmake --build ${BUILDDIR}