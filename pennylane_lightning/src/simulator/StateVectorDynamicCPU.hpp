#include <numeric>
#include <utility>
#include <vector>

#include "BitUtil.hpp"
#include "LinearAlgebra.hpp"

#include "Error.hpp"

#include <StateVectorCPU.hpp>

#include <iostream>

namespace Pennylane {
/**
 * @brief State-vector dynamic class.
 *
 * This class allocates and deallocates the number of qubits/wires dynamically,
 * and defines all operations to manipulate the statevector data for
 * quantum circuit simulation.
 *
 * @note This class introduces `WIRE_STATUS` so that operations can be applied
 * only on `ACTIVE` wires. `RELEASED` wires can be re-activated while `DISABLED`
 * wires are permanently destroyed.
 *
 */
template <class PrecisionT = double>
class StateVectorDynamicCPU
    : public StateVectorCPU<PrecisionT, StateVectorDynamicCPU<PrecisionT>> {
  public:
    using BaseType =
        StateVectorCPU<PrecisionT, StateVectorDynamicCPU<PrecisionT>>;

    using ComplexPrecisionT = std::complex<PrecisionT>;

    enum class WIRE_STATUS {
        ACTIVE,
        RELEASED,
        DISABLED,
    };

  private:
    std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>>
        data_;

    std::vector<WIRE_STATUS> wstatus_;
    // std::vector<long long> wmap_;

    template <class IIter, class OIter>
    OIter _move_data_elements(IIter first, size_t distance, OIter second) {
        *second++ = std::move(*first);
        for (size_t i = 1; i < distance; i++) {
            *second++ = std::move(*++first);
        }
        return second;
    }

    template <class IIter, class OIter>
    OIter _shallow_move_data_elements(IIter first, size_t distance,
                                      OIter second) {
        *second++ = std::move(*first);
        *first = ComplexPrecisionT{0, 0};
        for (size_t i = 1; i < distance; i++) {
            *second++ = std::move(*++first);
            *first = ComplexPrecisionT{0, 0};
        }
        return second;
    }

    void _scalar_mul_data(
        std::vector<ComplexPrecisionT,
                    Util::AlignedAllocator<ComplexPrecisionT>> &data,
        ComplexPrecisionT scalar) {
        std::transform(
            data.begin(), data.end(), data.begin(),
            [scalar](const ComplexPrecisionT &elem) { return elem * scalar; });
    }

    void _normalize_data(
        std::vector<ComplexPrecisionT,
                    Util::AlignedAllocator<ComplexPrecisionT>> &data) {
        _scalar_mul_data(
            data, std::complex<PrecisionT>{1, 0} /
                      std::sqrt(Util::squaredNorm(data.data(), data.size())));
    }

  public:
    /**
     * @brief Create a new statevector
     *
     * @param num_qubits Number of qubits
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    explicit StateVectorDynamicCPU(
        size_t num_qubits, Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model},
          data_{Util::exp2(num_qubits), ComplexPrecisionT{0.0, 0.0},
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        data_[0] = {1, 0};

        wstatus_.resize(num_qubits); // all of wires are ACTIVE.
    }
    /**
     * @brief Construct a statevector from another statevector
     *
     * @tparam OtherDerived A derived type of StateVectorCPU to use for
     * construction.
     * @param other Another statevector to construct the statevector from
     */
    template <class OtherDerived>
    explicit StateVectorDynamicCPU(
        const StateVectorCPU<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(),
                   other.memoryModel()),
          data_{other.getData(), other.getData() + other.getLength(),
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        wstatus_.resize(other.getNumQubits()); // all of wires are ACTIVE.
    }

    /**
     * @brief Construct a statevector from data pointer
     *
     * @param other_data Data pointer to construct the statvector from.
     * @param other_size Size of the data
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    StateVectorDynamicCPU(const ComplexPrecisionT *other_data,
                          size_t other_size,
                          Threading threading = Threading::SingleThread,
                          CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(Util::log2PerfectPower(other_size), threading, memory_model),
          data_{other_data, other_data + other_size,
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
        wstatus_.resize(Util::log2PerfectPower(other_size));
    }

    /**
     * @brief Construct a statevector from a data vector
     *
     * @tparam Alloc Allocator type of std::vector to use for constructing
     * statevector.
     * @param other Data to construct the statevector from
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    template <class Alloc>
    explicit StateVectorDynamicCPU(
        const std::vector<std::complex<PrecisionT>, Alloc> &other,
        Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorDynamicCPU(other.data(), other.size(), threading,
                                memory_model) {}

    StateVectorDynamicCPU(const StateVectorDynamicCPU &rhs) = default;
    StateVectorDynamicCPU(StateVectorDynamicCPU &&) noexcept = default;

    StateVectorDynamicCPU &operator=(const StateVectorDynamicCPU &) = default;
    StateVectorDynamicCPU &
    operator=(StateVectorDynamicCPU &&) noexcept = default;

    ~StateVectorDynamicCPU() = default;

    /**
     * @brief Get the status of a wire.
     *
     * @param wire Index of the wire.
     * @return WIRE_STATUS
     */
    [[nodiscard]] auto getWireStatus(size_t wire) -> WIRE_STATUS {
        assert(wire < wstatus_.size());
        return wstatus_[wire];
    }

    /**
     * @brief Get the total number of wires.
     */
    [[nodiscard]] auto getTotalNumWires() const -> size_t {
        return wstatus_.size();
    }

    /**
     * @brief Get the number of active wires.
     */
    [[nodiscard]] auto getNumActiveWires() const -> size_t {
        return std::count(wstatus_.begin(), wstatus_.end(),
                          WIRE_STATUS::ACTIVE);
    }

    /**
     * @brief Get the number of active wires up to `wire`.
     */
    [[nodiscard]] auto getNumActiveWires(size_t wire) const -> size_t {
        return std::count(wstatus_.begin(), wstatus_.begin() + wire,
                          WIRE_STATUS::ACTIVE);
    }

    /**
     * @brief Get the number of released wires.
     */
    [[nodiscard]] auto getNumReleasedWires() const -> size_t {
        return std::count(wstatus_.begin(), wstatus_.end(),
                          WIRE_STATUS::RELEASED);
    }

    /**
     * @brief Get the number of disabled wires.
     */
    [[nodiscard]] auto getNumDisabledWires() const -> size_t {
        return std::count(wstatus_.begin(), wstatus_.end(),
                          WIRE_STATUS::DISABLED);
    }

    /**
     * @brief Check if all of wires are ACTIVE.
     *
     * @param wires List of wires.
     * @return bool
     */
    [[nodiscard]] auto isActiveWires([[maybe_unused]] std::vector<size_t> wires)
        -> bool override {
        for (const auto &w : wires) {
            if (getWireStatus(w) != WIRE_STATUS::ACTIVE) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Compute the purity of the system after releasing (a qubit) `wire`.
     *
     * This traces out the complement of the wire for a more efficient
     * computation of the purity in O(N) with calculating the reduced density
     * matrix after tracing out the complement of qubit `wire`.
     *
     * @param wire Index of the wire.
     * @return ComplexPrecisionT
     */
    auto computeSystemPurity(size_t wire) -> ComplexPrecisionT {
        PL_ABORT_IF_NOT(getWireStatus(wire) == WIRE_STATUS::ACTIVE,
                        "Invalid wire status: The wire must be ACTIVE");

        const size_t sv_size = data_.size();
        const size_t local_wire_idx = getNumActiveWires(wire);

        // With `k` indexing the subsystem on n-1 qubits, we need to insert an
        // addtional bit into the index of the full state-vector at position
        // `wire`. These masks enable us to split the bits of the index `k` into
        // those above and below `wire`.
        const size_t lower_mask = (1 << local_wire_idx) - 1;
        const size_t upper_mask = sv_size - lower_mask - 1;

        // The resulting 2x2 reduced density matrix of the complement system to
        // qubit `wire`.
        std::vector<ComplexPrecisionT> rho(4, {0, 0});

        for (uint8_t i = 0; i < 2; i++) {
            for (uint8_t j = 0; j < 2; j++) {
                ComplexPrecisionT sum{0, 0};
                for (size_t k = 0; k < (sv_size >> 1); k++) {
                    size_t idx_wire_0 =
                        (/* upper_bits: */ (upper_mask & k) << 1) +
                        /* lower_bits: */ (lower_mask & k);
                    size_t idx_i = idx_wire_0 + (i << local_wire_idx);
                    size_t idx_j = idx_wire_0 + (j << local_wire_idx);

                    // This computes <00..i..00|psi><psi|00..j..00> on the first
                    // iteration, with the last iteration computing
                    // <11..i..11|psi><psi|11..j..11>.
                    sum += data_[idx_i] * std::conj(data_[idx_j]);
                }
                rho[2 * i + j] = sum;
            }
        }

        // Compute/Return the trace of rho**2
        return (rho[0] * rho[0]) + (ComplexPrecisionT{2, 0} * rho[1] * rho[2]) +
               (rho[3] * rho[3]);
    }

    /**
     * @brief Check the purity of a system after releasing/disabling `wire`.
     *
     * @param wire Index of the wire.
     * @param eps The comparing precision threshold.
     * @return bool
     */
    [[nodiscard]] auto
    isPureSystem(size_t wire,
                 double eps = std::numeric_limits<float>::epsilon() * 100)
        -> bool {
        ComplexPrecisionT purity = computeSystemPurity(wire);
        // std::cerr << "purity: " << purity << std::endl;
        return (std::abs(1.0 - purity.real()) < eps) && (purity.imag() < eps);
    }

    /**
     * @brief Add a new wire or re-use a released one
     *
     * @param wire If < 0, it first tries to reuse the smallest released
     * wire, in case of the failure, it adds a new one at the end of the
     * list of `ACTIVE` wires. If >= 0, then the status of the `wire` must
     * be `RELEASED`, otherwise it throws LightningException. @note the value
     * of wire = -1 by default.
     *
     * @return It updates the state-vector and the number of qubits,
     * and returns wire of the activated wire.
     */
    auto activateWire(long wire = -1) -> size_t {
        assert(wire < static_cast<long>(wstatus_.size()));
        size_t next_idx;
        bool in_middle = true;
        if (wire < 0) {
            auto released_wire = std::find(wstatus_.begin(), wstatus_.end(),
                                           WIRE_STATUS::RELEASED);
            if (released_wire == std::end(wstatus_)) {
                next_idx = wstatus_.size();
                wstatus_.push_back(WIRE_STATUS::ACTIVE);
                in_middle = false;
            } else {
                next_idx = released_wire - wstatus_.begin();
                *released_wire = WIRE_STATUS::ACTIVE;
            }
        } else {
            PL_ABORT_IF_NOT(getWireStatus(wire) == WIRE_STATUS::RELEASED,
                            "The wire must be released before activation.");
            next_idx = wire;
        }

        data_.resize(data_.size() << 1);
        if (in_middle) {
            const size_t local_wire_idx = getNumActiveWires(wire);
            const size_t distance = 1ul << local_wire_idx;
            ComplexPrecisionT *second = &data_[data_.size() >> 1];
            second -= distance;
            for (auto first = data_.end() - distance; second >= &data_[0];
                 second -= distance, first -= distance << 1) {
                _shallow_move_data_elements(second, distance, first);
            }
        }

        this->setNumQubits(this->getNumQubits() + 1);
        return next_idx;
    }

    /**
     * @brief Release an `ACTIVE` wire
     *
     * @param wire Index of the wire to be released.
     *
     * @note This updates the state-vector and reduces the number
     * of qubits. But does nothing if the wire's status is either
     * `RELEASED` or `DISABLED`.
     */
    void releaseWire(size_t wire) {
        const auto status = getWireStatus(wire);
        if (status == WIRE_STATUS::RELEASED ||
            status == WIRE_STATUS::DISABLED) {
            return;
        }

        PL_ABORT_IF_NOT(
            isPureSystem(wire),
            "Invalid wire: "
            "The state-vector must remain pure after releasing a wire")

        // To catch cases with multiple released/disabled wires,
        const size_t local_wire_idx = getNumActiveWires(wire);

        // if it's either |0> or |1> but not both,
        const long distance = 1l << local_wire_idx;
        auto second = data_.begin();
        for (auto first = second + distance; first < data_.end();
             first += distance << 1, second += distance) {
            _move_data_elements(first, distance, second);
        }

        data_.resize(data_.size() >> 1);
        this->setNumQubits(this->getNumQubits() - 1);
        wstatus_[wire] = WIRE_STATUS::RELEASED;
    }

    /**
     * @brief Disable an `ACTIVE` or `RELEASED` wire
     *
     * @param wire Index of the wire to be disabled.
     *
     * @note This updates the state-vector and reduces the number
     * of qubits. But does nothing if the wire's status is `DISABLED`.
     *
     * @note The `DISABLED` wires cannot be activated.
     */
    void disableWire(size_t wire) {
        const auto status = getWireStatus(wire);
        if (status == WIRE_STATUS::DISABLED) {
            return;
        } else if (status == WIRE_STATUS::RELEASED) {
            wstatus_[wire] = WIRE_STATUS::DISABLED;
            return;
        }

        PL_ABORT_IF_NOT(
            isPureSystem(wire),
            "Invalid wire: "
            "The state-vector must remain pure after disabling a wire")

        // To catch cases with multiple released/disabled wires,
        const size_t local_wire_idx = getNumActiveWires(wire);

        const long distance = 1l << local_wire_idx;
        auto second = data_.begin();
        for (auto first = second + distance; first < data_.end();
             first += distance << 1, second += distance) {
            _move_data_elements(first, distance, second);
        }

        data_.resize(data_.size() >> 1);
        this->setNumQubits(this->getNumQubits() - 1);
        wstatus_[wire] = WIRE_STATUS::DISABLED;
    }

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_.data();
    }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector()
        -> std::vector<ComplexPrecisionT,
                       Util::AlignedAllocator<ComplexPrecisionT>> & {
        return data_;
    }

    [[nodiscard]] auto getDataVector() const
        -> const std::vector<ComplexPrecisionT,
                             Util::AlignedAllocator<ComplexPrecisionT>> & {
        return data_;
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @tparam Alloc Allocator type of std::vector to use for updating data.
     * @param new_data std::vector contains data.
     */
    template <class Alloc>
    void updateData(const std::vector<ComplexPrecisionT, Alloc> &new_data) {
        assert(data_.size() == new_data.size());
        std::copy(new_data.data(), new_data.data() + new_data.size(),
                  data_.data());
    }

    Util::AlignedAllocator<ComplexPrecisionT> allocator() const {
        return data_.get_allocator();
    }
};

} // namespace Pennylane