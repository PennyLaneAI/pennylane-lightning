#include "AdjointDiff.hpp"
#include "JacobianTape.hpp"
#include "Kokkos_Sparse.hpp"
#include "Macros.hpp"
#include "Measures.hpp"
#include "Observables.hpp"
#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"

// ******************************************************************** //
// Avoid redundant builds of classes explicitly instantiated elsewhere
// ******************************************************************** //

extern template class Pennylane::Algorithms::NamedObs<float>;
extern template class Pennylane::Algorithms::NamedObs<double>;
extern template class Pennylane::Algorithms::HermitianObs<float>;
extern template class Pennylane::Algorithms::HermitianObs<double>;
extern template class Pennylane::Algorithms::TensorProdObs<float>;
extern template class Pennylane::Algorithms::TensorProdObs<double>;
extern template class Pennylane::Algorithms::Hamiltonian<float>;
extern template class Pennylane::Algorithms::Hamiltonian<double>;

extern template class Pennylane::Algorithms::OpsData<float>;
extern template class Pennylane::Algorithms::OpsData<double>;

extern template class Pennylane::Algorithms::JacobianData<float>;
extern template class Pennylane::Algorithms::JacobianData<double>;

extern template class Pennylane::StateVectorManagedCPU<float>;
extern template class Pennylane::StateVectorManagedCPU<double>;

extern template class Pennylane::StateVectorRawCPU<float>;
extern template class Pennylane::StateVectorRawCPU<double>;

extern template class Pennylane::Measures<float,
                                          Pennylane::StateVectorRawCPU<float>>;
extern template class Pennylane::Measures<double,
                                          Pennylane::StateVectorRawCPU<double>>;

// ************************************************************** //