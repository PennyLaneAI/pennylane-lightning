#include <numeric>
#include <utility>
#include <vector>

#include "BitUtil.hpp"
#include "Error.hpp"

#include <StateVectorCPU.hpp>

#include <iostream>

namespace Pennylane {

// template<class T = size_t>
// class Wires {
//     private:
//         std::vector<T> map_;
//         T num_wires_;

//     public:
//         static constexpr T invalid_wire_id = std::numeric_limits<T>::max();

//     public:
//         Wires(T num_wires) : num_wires_(num_wires),
//             std::iota(map_.begin(), map_.end(), 0) {}
//         ~Wires() = default;

//         auto getNumWires() const -> T {
//             return num_wires_;
//         }

//         auto getWireId(T w) const -> T {
//             assert(w < num_wires_);
//             return map_.at(w);
//         }

//         auto isActiveWire(T w) -> bool {
//             assert(w < num_wires_);
//             return map_[w] != invalid_wire_id;
//         }

//         void updateWire(T w, T new_id) {
//             if(w < num_wires_) {
//                 map_[w] = new_id;
//             }
//         }
// };

/**
 * @brief State-vector dynamic class.
 *
 * This class allocates and deallocates dynamically, and defines all operations
 * to manipulate the statevector data for quantum circuit simulation.
 *
 * @note
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
    std::vector<WIRE_STATUS> wmap_;

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

        wmap_.resize(num_qubits); // all of wires are ACTIVE.
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
        wmap_.resize(other.getNumQubits()); // all of wires are ACTIVE.
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
        wmap_.resize(Util::log2PerfectPower(other_size));
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
     * @param wire The index of wire.
     * @return WIRE_STATUS
     */
    [[nodiscard]] auto getWireStatus(size_t wire) -> WIRE_STATUS {
        assert(wire < wmap_.size());
        return wmap_[wire];
    }

    /**
     * @brief Get the total number of wires.
     */
    [[nodiscard]] auto getTotalNumWires() const -> size_t {
        return wmap_.size();
    }

    /**
     * @brief Get the number of active wires.
     */
    [[nodiscard]] auto getNumActiveWires() const -> size_t {
        return std::count(wmap_.begin(), wmap_.end(), WIRE_STATUS::ACTIVE);
    }

    /**
     * @brief Get the number of released wires.
     */
    [[nodiscard]] auto getNumReleasedWires() const -> size_t {
        return std::count(wmap_.begin(), wmap_.end(), WIRE_STATUS::RELEASED);
    }

    /**
     * @brief Get the number of disabled wires.
     */
    [[nodiscard]] auto getNumDisabledWires() const -> size_t {
        return std::count(wmap_.begin(), wmap_.end(), WIRE_STATUS::DISABLED);
    }

    // TODO(ali): remove after debugging
    void setWireStatus_debug(size_t wire, WIRE_STATUS status) {
        assert(wire < wmap_.size());
        wmap_[wire] = status;
    }

    /**
     * @brief Check if all of wires are ACTIVE.
     *
     * @param wires The list of wires.
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
     * @brief Add a new wire or re-use a released wire
     *
     * @param index If < 0, it first tries to reuse the smallest released
     * wire, in case of the failure, it adds a new one at the end of the
     * list of `ACTIVE` wires. If >= 0, then the status of the index-wire must
     * be `RELEASED`, otherwise it throws LightningException. @note the value
     * of index = -1 by default.
     * @return It updates the state-vector and the number of qubits,
     * and returns index of the activated wire.
     */
    auto activateWire(long index = -1) -> size_t {
        assert(index < static_cast<long>(wmap_.size()));
        size_t next_idx;
        bool in_middle = true;
        if (index < 0) {
            auto released_wire =
                std::find(wmap_.begin(), wmap_.end(), WIRE_STATUS::RELEASED);
            if (released_wire == std::end(wmap_)) {
                next_idx = wmap_.size();
                wmap_.push_back(WIRE_STATUS::ACTIVE);
                in_middle = false;
            } else {
                next_idx = released_wire - wmap_.begin();
                *released_wire = WIRE_STATUS::ACTIVE;
            }
        } else {
            PL_ABORT_IF_NOT(getWireStatus(index) == WIRE_STATUS::RELEASED,
                            "The wire must be released before activation.");
            next_idx = index;
        }

        data_.resize(data_.size() << 1);
        if (in_middle) {
            const size_t distance = 1ul << next_idx;
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
     * @param index The index of wire to be released.
     *
     * @note This updates the state-vector and reduces the number
     * of qubits. But does nothing if the wire's status is either
     * `RELEASED` or `DISABLED`.
     */
    void releaseWire(size_t index) {
        const auto status = getWireStatus(index);
        if (status == WIRE_STATUS::RELEASED ||
            status == WIRE_STATUS::DISABLED) {
            return;
        }

        // if it's either |0> or |1> but not both,
        const long distance = 1l << index;
        auto second = data_.begin();
        for (auto first = second + distance; first < data_.end();
             first += distance << 1, second += distance) {
            _move_data_elements(first, distance, second);
        }

        data_.resize(data_.size() >> 1);
        this->setNumQubits(this->getNumQubits() - 1);
        wmap_[index] = WIRE_STATUS::RELEASED;
    }

    /**
     * @brief Disable an `ACTIVE` or `RELEASED` wire
     *
     * @param index The index of wire to be disabled.
     *
     * @note This updates the state-vector and reduces the number
     * of qubits. But does nothing if the wire's status is `DISABLED`.
     *
     * @note The `DISABLED` wires cannot be activated.
     */
    void disableWire(size_t index) {
        const auto status = getWireStatus(index);
        if (status == WIRE_STATUS::DISABLED) {
            return;
        } else if (status == WIRE_STATUS::RELEASED) {
            wmap_[index] = WIRE_STATUS::DISABLED;
            return;
        }

        const long distance = 1l << index;
        auto second = data_.begin();
        for (auto first = second + distance; first < data_.end();
             first += distance << 1, second += distance) {
            _move_data_elements(first, distance, second);
        }

        data_.resize(data_.size() >> 1);
        this->setNumQubits(this->getNumQubits() - 1);
        wmap_[index] = WIRE_STATUS::DISABLED;
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