// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

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

namespace Pennylane::TraceUtil {

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

} // namespace Pennylane::TraceUtil