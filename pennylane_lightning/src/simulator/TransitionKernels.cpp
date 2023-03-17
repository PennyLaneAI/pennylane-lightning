#include "TransitionKernels.hpp"

// explicit instantiation
template class Pennylane::TransitionKernel<float>;
template class Pennylane::TransitionKernel<double>;
template class Pennylane::LocalTransitionKernel<float>;
template class Pennylane::LocalTransitionKernel<double>;
template class Pennylane::NonZeroRandomTransitionKernel<float>;
template class Pennylane::NonZeroRandomTransitionKernel<double>;
