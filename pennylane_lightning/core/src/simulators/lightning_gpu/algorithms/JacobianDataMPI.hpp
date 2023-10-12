#pragma once
#include "MPIManager.hpp"
#include "DevTag.hpp"
#include "JacobianData.hpp"

// using namespace Pennylane;
/// @cond DEV
namespace {
using Pennylane::Observables::Observable;
using namespace Pennylane::LightningGPU::MPI;
} // namespace
/// @endcond

namespace Pennylane::Algorithms {
template <class StateVectorT>
class JacobianDataMPI final : public JacobianData<StateVectorT> {
  private:
    using BaseType = JacobianData<StateVectorT>;
    const size_t numGlobalQubits_;
    const size_t numLocalQubits_;

    MPIManager mpi_manager_;
    const DevTag<int> dev_tag_;

  public:
    JacobianDataMPI(const JacobianDataMPI &) = default;
    JacobianDataMPI(JacobianDataMPI &&) noexcept = default;
    JacobianDataMPI &operator=(const JacobianDataMPI &) = default;
    JacobianDataMPI &operator=(JacobianDataMPI &&) noexcept = default;
    virtual ~JacobianDataMPI() = default;

    /**
     * @brief Construct a JacobianData object
     *
     * @param num_params Number of parameters in the Tape.
     * @param sv Referemce to the statevector.
     * @param obs Observables for which to calculate Jacobian.
     * @param ops Operations used to create given state.
     * @param trainP Sorted list of parameters participating in Jacobian
     * computation.
     *
     * @rst
     * Each value :math:`i` in trainable params means that
     * we want to take a derivative respect to the :math:`i`-th operation.
     *
     * Further note that ``ops`` does not contain state preparation operations
     * (e.g. StatePrep) or Hamiltonian coefficients.
     * @endrst
     */
    JacobianDataMPI(size_t num_params, const StateVectorT &sv,
                    std::vector<std::shared_ptr<Observable<StateVectorT>>> obs,
                    OpsData<StateVectorT> ops, std::vector<size_t> trainP)
        : JacobianData<StateVectorT>(num_params, sv.getLength(), sv.getData(),
                                     obs, ops, trainP),
          numGlobalQubits_(sv.getNumGlobalQubits()),
          numLocalQubits_(sv.getNumLocalQubits()),
          mpi_manager_(sv.getMPIManager()),
          dev_tag_(sv.getDataBuffer().getDevTag()) {
        /* When the Hamiltonian has parameters, trainable parameters include
         * these. We explicitly ignore them. */
        mpi_manager_.Barrier();
    }

    /**
     * @brief Get MPI manager
     */
    auto getMPIManager() const { return mpi_manager_; }

    /**
     * @brief Get DevTag manager
     */
    auto getDevTag() const { return dev_tag_; };

    /**
     * @brief Get the number of wires distributed across devices.
     */
    auto getNumGlobalQubits() const -> size_t { return numGlobalQubits_; }

    /**
     * @brief Get the number of wires within the local devices.
     */
    auto getNumLocalQubits() const -> size_t { return numLocalQubits_; }
};
} // namespace Pennylane::Algorithms
