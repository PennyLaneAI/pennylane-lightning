#pragma once

namespace Pennylane::LightningQubit::Gates::Pragmas {

#ifdef PL_LQ_KERNEL_OMP
#define LOOP_PRAGMA #pragma omp parallel for
#else
#define LOOP_PRAGMA
#endif

}; // namespace Pennylane::LightningQubit::Gates::Pragmas