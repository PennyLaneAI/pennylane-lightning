#pragma once

namespace Pennylane::LightningQubit::Gates::Pragmas {

#ifdef PL_LQ_KERNEL_OMP
#define LOOP_PARALLEL #pragma omp parallel for
#define LOOP_SIMD #pragma omp simd
#else
#define LOOP_PARALLEL
#define LOOP_SIMD
#endif

}; // namespace Pennylane::LightningQubit::Gates::Pragmas