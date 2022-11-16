#pragma once

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"

namespace Pennylane {

enum class TransitionKernelType { Local, NonZeroRandom };

/**
 * @brief Parent class to define interface for Transition Kernel
 *
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <typename fp_t> class TransitionKernel {
  public:
    //  outputs the next state and the qratio
    virtual std::pair<size_t, fp_t> operator()(size_t) = 0;
};

/**
 * @brief Transition Kernel for a 'SpinFlip' local transition between states
 *
 * This class implements a local transition kernel for a spin flip operation.
 * It goes about this by generating a random qubit site and then generating
 * a random number to determine the new bit at that qubit site.
 * @tparam fp_t Floating point precision of underlying measurements.
 */
template <typename fp_t>
class LocalTransitionKernel : public TransitionKernel<fp_t> {
  private:
    size_t num_qubits_;
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_num_qubits_;
    std::uniform_int_distribution<size_t> distrib_binary_;

  public:
    explicit LocalTransitionKernel(size_t num_qubits)
        : num_qubits_(num_qubits), gen_(std::mt19937(rd_())),
          distrib_num_qubits_(
              std::uniform_int_distribution<size_t>(0, num_qubits - 1)),
          distrib_binary_(std::uniform_int_distribution<size_t>(0, 1)) {}

    std::pair<size_t, fp_t> operator()(size_t s1) final {
        size_t qubit_site = distrib_num_qubits_(gen_);
        size_t qubit_value = distrib_binary_(gen_);
        size_t current_bit = ((unsigned)s1 >> (unsigned)qubit_site) & 1u;

        if (qubit_value == current_bit) {
            return std::pair<size_t, fp_t>(s1, 1);
        }
        if (current_bit == 0) {
            return std::pair<size_t, fp_t>(s1 + std::pow(2, qubit_site), 1);
        }
        return std::pair<size_t, fp_t>(s1 - std::pow(2, qubit_site), 1);
    }
};

/**
 * @brief Transition Kernel for a random transition between non-zero states
 *
 * This class randomly transitions between states that have nonzero probability.
 * To determine the states with non-zero probability we have O(2^num_qubits)
 * overhead. Despite this, this method is still fast. This transition kernel
 * can sample even GHZ states.
 */
template <typename fp_t>
class NonZeroRandomTransitionKernel : public TransitionKernel<fp_t> {
  private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> distrib_;
    size_t sv_length_;
    std::vector<size_t> non_zeros_;

  public:
    NonZeroRandomTransitionKernel(const std::complex<fp_t> *sv,
                                  size_t sv_length, fp_t min_error) {
        auto data = sv;
        sv_length_ = sv_length;

        // find nonzero candidates
        for (size_t i = 0; i < sv_length_; i++) {
            if (std::fabs(data[i].real()) > min_error ||
                std::fabs(data[i].imag()) > min_error) {
                non_zeros_.push_back(i);
            }
        }
        gen_ = std::mt19937(rd_());
        distrib_ =
            std::uniform_int_distribution<size_t>(0, non_zeros_.size() - 1);
    }

    std::pair<size_t, fp_t> operator()([[maybe_unused]] size_t s1) final {
        auto s2 = distrib_(gen_);
        return std::pair<size_t, fp_t>(non_zeros_[s2], 1);
    }
};

/**
 * @brief Factory function to create a transition kernel
 *
 * @param kernel_type Type of transition kernel to create
 * @param sv pointer to the statevector data
 * @param num_qubits number of qubits
 * @tparam fp_t Floating point precision of underlying measurements.
 * @return std::unique_ptr of the transition kernel
 */
template <typename fp_t>
std::unique_ptr<TransitionKernel<fp_t>>
kernel_factory(const TransitionKernelType kernel_type,
               const std::complex<fp_t> *sv, size_t num_qubits) {

    auto sv_length = Util::exp2(num_qubits);
    if (kernel_type == TransitionKernelType::Local) {
        return std::unique_ptr<TransitionKernel<fp_t>>(
            new NonZeroRandomTransitionKernel<fp_t>(
                sv, sv_length, std::numeric_limits<fp_t>::epsilon()));
    }
    return std::unique_ptr<TransitionKernel<fp_t>>(
        new LocalTransitionKernel<fp_t>(num_qubits));
}
} // namespace Pennylane
