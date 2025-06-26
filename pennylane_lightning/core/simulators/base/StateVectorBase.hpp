// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file StateVectorBase.hpp
 * Defines a base class for all simulators.
 */

#pragma once

/// @cond DEV
// Required for compilation with MSVC
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES // for C++
#endif
/// @endcond

#include <complex>

#include <perfetto.h>

#include <memory>
#include <chrono>
#include <fstream>
#include <thread>

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("binding")
        .SetDescription("Bindings events"),
    perfetto::Category("statevector")
        .SetDescription("Statevector events"),
    perfetto::Category("measurement")
        .SetDescription("Measurement events"),);

namespace Pennylane {



void InitializePerfetto() {
perfetto::TracingInitArgs args;
// The backends determine where trace events are recorded. For this example we
// are going to use the in-process tracing service, which only includes in-app
// events.
args.backends = perfetto::kInProcessBackend;

perfetto::Tracing::Initialize(args);
perfetto::TrackEvent::Register();
}

std::unique_ptr<perfetto::TracingSession> StartTracing() {
// The trace config defines which types of data sources are enabled for
// recording. In this example we just need the "track_event" data source,
// which corresponds to the TRACE_EVENT trace points.
perfetto::TraceConfig cfg;
cfg.add_buffers()->set_size_kb(1024);
auto* ds_cfg = cfg.add_data_sources()->mutable_config();
ds_cfg->set_name("track_event");
perfetto::protos::gen::TrackEventConfig te_cfg;
te_cfg.add_disabled_categories("*");
te_cfg.add_enabled_categories("rendering");
ds_cfg->set_track_event_config_raw(te_cfg.SerializeAsString());

auto tracing_session = perfetto::Tracing::NewTrace();
tracing_session->Setup(cfg);
tracing_session->StartBlocking();
return tracing_session;
}

void StopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session) {
// Make sure the last event is closed for this example.
perfetto::TrackEvent::Flush();

// Stop tracing and read the trace data.
tracing_session->StopBlocking();
std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

// Write the result into a file.
// Note: To save memory with longer traces, you can tell Perfetto to write
// directly into a file by passing a file descriptor into Setup() above.
std::ofstream output;
output.open("example.pftrace", std::ios::out | std::ios::binary);
output.write(&trace_data[0], std::streamsize(trace_data.size()));
output.close();
PERFETTO_LOG(
    "Trace written in example.pftrace file. To read this trace in "
    "text form, run `./tools/traceconv text example.pftrace`");
}


/**
 * @brief State-vector base class.
 *
 * This class combines a data array managed by a derived class (CRTP) and
 * implementations of gate operations. The bound data is assumed to be complex,
 * and is required to be in either 32-bit (64-bit `complex<float>`) or
 * 64-bit (128-bit `complex<double>`) floating point representation.
 * As this is the base class, we do not add default template arguments.
 *
 * @tparam T Floating point precision of underlying statevector data.
 * @tparam Derived Type of a derived class
 */
template <class PrecisionT, class Derived> class StateVectorBase {
  protected:
    std::size_t num_qubits_{0};

  public:
    /**
     * @brief StateVector complex precision type.
     */
    using ComplexT = std::complex<PrecisionT>;

    /**
     * @brief Constructor used by derived classes.
     *
     * @param num_qubits Number of qubits
     */
    explicit StateVectorBase(std::size_t num_qubits)
        : num_qubits_{num_qubits} {}

    /**
     * @brief Get the number of qubits represented by the statevector data.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    /**
     * @brief Get the total number of qubits of the simulated system.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getTotalNumQubits() const -> std::size_t {
        return num_qubits_;
    }

    /**
     * @brief Get the size of the statevector
     *
     * @return The size of the statevector
     */
    [[nodiscard]] std::size_t getLength() const {
        return static_cast<std::size_t>(exp2(num_qubits_));
    }

    /**
     * @brief Get the data pointer of the statevector
     *
     * @return A pointer to the statevector data
     */
    [[nodiscard]] inline auto getData() -> decltype(auto) {
        return static_cast<Derived *>(this)->getData();
    }

    [[nodiscard]] inline auto getData() const -> decltype(auto) {
        return static_cast<const Derived *>(this)->getData();
    }

    /**
     * @brief Apply a single gate to the state-vector.
     *
     * @param opName Gate's name.
     * @param wires Wires to apply gate to.
     * @param adjoint Indicates whether to use adjoint of gate.
     * @param params Optional parameter list for parametric gates.
     */
    inline void applyOperation(const std::string &opName,
                               const std::vector<std::size_t> &wires,
                               bool adjoint = false,
                               const std::vector<PrecisionT> &params = {}) {
        return static_cast<Derived *>(this)->applyOperation(opName, wires,
                                                            adjoint, params);
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     * @param ops_params Optional parameter data for index matched gates.
     */
    void
    applyOperations(const std::vector<std::string> &ops,
                    const std::vector<std::vector<std::size_t>> &ops_wires,
                    const std::vector<bool> &ops_adjoint,
                    const std::vector<std::vector<PrecisionT>> &ops_params) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF(
            numOperations != ops_wires.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_adjoint.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        PL_ABORT_IF(
            numOperations != ops_params.size(),
            "Invalid arguments: number of operations, wires, inverses, and "
            "parameters must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            this->applyOperation(ops[i], ops_wires[i], ops_adjoint[i],
                                 ops_params[i]);
        }
    }

    /**
     * @brief Apply multiple gates to the state-vector.
     *
     * @param ops Vector of gate names to be applied in order.
     * @param ops_wires Vector of wires on which to apply index-matched gate
     * name.
     * @param ops_adjoint Indicates whether gate at matched index is to be
     * inverted.
     */
    void applyOperations(const std::vector<std::string> &ops,
                         const std::vector<std::vector<std::size_t>> &ops_wires,
                         const std::vector<bool> &ops_adjoint) {
        const std::size_t numOperations = ops.size();
        PL_ABORT_IF_NOT(
            numOperations == ops_wires.size(),
            "Invalid arguments: number of operations, wires, and inverses "
            "must all be equal");
        PL_ABORT_IF_NOT(
            numOperations == ops_adjoint.size(),
            "Invalid arguments: number of operations, wires and inverses"
            "must all be equal");
        for (std::size_t i = 0; i < numOperations; i++) {
            this->applyOperation(ops[i], ops_wires[i], ops_adjoint[i], {});
        }
    }

    /**
     * @brief Apply a single generator to the state-vector.
     *
     * @param opName Name of generator to apply.
     * @param wires Wires the generator applies to.
     * @param adjoint Indicates whether to use adjoint of operator.
     */
    inline auto applyGenerator(const std::string &opName,
                               const std::vector<std::size_t> &wires,
                               bool adjoint = false) -> PrecisionT {
        return static_cast<Derived *>(this)->applyGenerator(opName, wires,
                                                            adjoint);
    }

    /**
     * @brief Apply a given matrix directly to the statevector using a
     * raw matrix pointer vector.
     *
     * @param matrix Pointer to the array data (in row-major format).
     * @param wires Wires to apply gate to.
     * @param inverse Indicate whether inverse should be taken.
     */
    inline void applyMatrix(const ComplexT *matrix,
                            const std::vector<std::size_t> &wires,
                            bool inverse = false) {
        return static_cast<Derived *>(this)->applyMatrix(matrix, wires,
                                                         inverse);
    }
};

} // namespace Pennylane
