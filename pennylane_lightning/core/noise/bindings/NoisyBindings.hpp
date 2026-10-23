// Copyright 2026 Xanadu Quantum Technologies Inc.

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
 * @file NoisyBindings.hpp
 * Nanobind helpers for the Python modules of the noisy Kokkos backends
 * (lightning_tensornet). A backend state implements the state API used by
 * the Catalyst NoisySimulator, so the common methods are registered by a
 * single template.
 */
#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "Kokkos_Core.hpp"

#include "GateMatrices.hpp"
#include "NoiseModel.hpp"
#include "ObservableTerms.hpp"

namespace Pennylane::LightningNoisy::NanoBindings {

namespace nb = nanobind;

template <class T>
using ArrayC =
    nb::ndarray<const std::complex<T>, nb::c_contig, nb::device::cpu>;

template <class T> CVec<T> toCVec(const ArrayC<T> &a) {
    const auto *p = a.data();
    return CVec<T>(p, p + a.size());
}

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> toNumpy(std::vector<T> &&v) {
    auto *heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<1>>(heap->data(), {heap->size()},
                                                  owner);
}

template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<2>>
toNumpy2(std::vector<T> &&v, std::size_t rows, std::size_t cols) {
    auto *heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void *p) noexcept {
        delete static_cast<std::vector<T> *>(p);
    });
    return nb::ndarray<nb::numpy, T, nb::ndim<2>>(heap->data(), {rows, cols},
                                                  owner);
}

/**
 * @brief Kokkos settings from a Python dict (keys follow
 * Kokkos::InitializationSettings: num_threads, device_id, map_device_id_by,
 * disable_warnings, print_configuration, tune_internals, tools_libs,
 * tools_help, tools_args).
 */
inline Kokkos::InitializationSettings kokkosSettings(const nb::dict &d) {
    Kokkos::InitializationSettings s;
    for (auto [k, v] : d) {
        const std::string key = nb::cast<std::string>(k);
        if (key == "num_threads") {
            s.set_num_threads(nb::cast<int>(v));
        } else if (key == "device_id") {
            s.set_device_id(nb::cast<int>(v));
        } else if (key == "map_device_id_by") {
            s.set_map_device_id_by(nb::cast<std::string>(v));
        } else if (key == "disable_warnings") {
            s.set_disable_warnings(nb::cast<bool>(v));
        } else if (key == "print_configuration") {
            s.set_print_configuration(nb::cast<bool>(v));
        } else if (key == "tune_internals") {
            s.set_tune_internals(nb::cast<bool>(v));
        } else if (key == "tools_libs") {
            s.set_tools_libs(nb::cast<std::string>(v));
        } else if (key == "tools_help") {
            s.set_tools_help(nb::cast<bool>(v));
        } else if (key == "tools_args") {
            s.set_tools_args(nb::cast<std::string>(v));
        } else {
            throw nb::value_error(("Unknown Kokkos setting: " + key).c_str());
        }
    }
    return s;
}

/**
 * @brief Register Kokkos helpers (initialize/finalize/configuration).
 */
inline void registerKokkosInfo(nb::module_ &m) {
    m.def("kokkos_is_initialized", []() { return Kokkos::is_initialized(); });
    m.def("kokkos_is_finalized", []() { return Kokkos::is_finalized(); });
    m.def(
        "print_configuration",
        []() {
            std::ostringstream buffer;
            Kokkos::print_configuration(buffer, true);
            return buffer.str();
        },
        "Kokkos configuration.");
    m.def("kokkos_execution_space",
          []() { return std::string(Kokkos::DefaultExecutionSpace::name()); });
}

/**
 * @brief Native noise model, owned per backend module (the holder type is
 * parametrised by the state class so that two backend modules loaded in the
 * same nanobind domain never register the same C++ type).
 */
template <class StateT> struct NativeNoiseModel {
    using PrecisionT = typename StateT::PrecisionT;
    NoiseModel<PrecisionT> model;
    explicit NativeNoiseModel(const std::string &spec) : model(spec) {}
};

using NoiseList = std::vector<std::pair<std::size_t, std::vector<std::size_t>>>;

template <class StateT>
void registerNoiseModel(nb::module_ &m, const std::string &name) {
    using NM = NativeNoiseModel<StateT>;
    auto toList = [](const std::vector<ResolvedNoise> &v) {
        NoiseList out;
        out.reserve(v.size());
        for (const auto &r : v) {
            out.emplace_back(r.channel, r.wires);
        }
        return out;
    };
    nb::class_<NM>(m, name.c_str())
        .def(nb::init<const std::string &>(), nb::arg("spec"))
        .def_prop_ro("num_channels",
                     [](const NM &self) { return self.model.numChannels(); })
        .def_prop_ro(
            "has_readout_noise",
            [](const NM &self) { return self.model.hasReadoutNoise(); })
        .def("empty", [](const NM &self) { return self.model.empty(); })
        .def(
            "lookup_gate",
            [toList](const NM &self, const std::string &gate,
                     const std::vector<std::size_t> &wires) {
                std::vector<ResolvedNoise> before;
                std::vector<ResolvedNoise> after;
                self.model.lookupGate(gate, wires, before, after);
                return std::make_pair(toList(before), toList(after));
            },
            nb::arg("gate"), nb::arg("wires"),
            "Return (before, after) lists of (channel, wires) for a gate.")
        .def(
            "lookup_readout",
            [toList](const NM &self, const std::string &kind,
                     const std::vector<std::size_t> &wires) {
                std::vector<ResolvedNoise> out;
                self.model.lookupReadout(kind, wires, out);
                return toList(out);
            },
            nb::arg("kind"), nb::arg("wires"))
        .def(
            "kraus",
            [](const NM &self, std::size_t i) {
                const auto &ch = self.model.channel(i);
                std::vector<CVec<typename NM::PrecisionT>> out = ch.kraus;
                return out;
            },
            nb::arg("channel"), "Kraus operators of a channel (flattened).");
}

/**
 * @brief Build ObsTerms from the flattened representation used by the
 * Python bindings.
 */
template <class T>
ObsTerms<T> makeTerms(const std::vector<std::complex<T>> &coeffs,
                      const std::vector<std::vector<std::size_t>> &wires,
                      const std::vector<std::string> &paulis,
                      const std::vector<ArrayC<T>> &matrices) {
    const std::size_t n = coeffs.size();
    if (wires.size() != n || paulis.size() != n || matrices.size() != n) {
        throw nb::value_error("Observable term lists differ in length.");
    }
    ObsTerms<T> terms(n);
    for (std::size_t k = 0; k < n; k++) {
        terms[k].coeff = coeffs[k];
        terms[k].wires = wires[k];
        if (matrices[k].size() != 0) {
            if (matrices[k].size() != pow2(2 * wires[k].size())) {
                throw nb::value_error(
                    "Observable matrix size does not match its wires.");
            }
            terms[k].dense = true;
            terms[k].matrix = toCVec<T>(matrices[k]);
        } else {
            if (paulis[k].size() != wires[k].size()) {
                throw nb::value_error("Pauli word and wires differ in length.");
            }
            terms[k].paulis = paulis[k];
        }
    }
    return terms;
}

/**
 * @brief Register the methods shared by every noisy backend state.
 *
 * @tparam StateT Backend state class (e.g. MPSKokkos).
 * @param cls Nanobind class to which the methods are added.
 */
template <class StateT, class ClassT> void registerCommonState(ClassT &cls) {
    using T = typename StateT::PrecisionT;
    using C = std::complex<T>;
    using NM = NativeNoiseModel<StateT>;

    cls.def_prop_ro("num_qubits", &StateT::getNumQubits)
        .def("reset_state", &StateT::resetState)
        .def(
            "set_basis_state",
            [](StateT &s, const std::vector<std::size_t> &state,
               const std::vector<std::size_t> &wires) {
                s.setBasisState(state, wires);
            },
            nb::arg("state"), nb::arg("wires"))
        .def(
            "set_pure_state",
            [](StateT &s, const ArrayC<T> &psi,
               const std::vector<std::size_t> &wires) {
                s.setPureState(psi.data(), psi.size(), wires);
            },
            nb::arg("state"), nb::arg("wires"))
        .def(
            "apply_operation",
            [](StateT &s, const std::string &name,
               const std::vector<std::size_t> &wires, bool inverse,
               const std::vector<T> &params,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values) {
                s.applyOperation(name, controlled_wires, controlled_values,
                                 wires, inverse, params);
            },
            nb::arg("name"), nb::arg("wires"), nb::arg("inverse") = false,
            nb::arg("params") = std::vector<T>{},
            nb::arg("controlled_wires") = std::vector<std::size_t>{},
            nb::arg("controlled_values") = std::vector<bool>{})
        .def(
            "apply_pauli_rot",
            [](StateT &s, const std::vector<std::size_t> &wires, bool inverse,
               const std::vector<T> &params, const std::string &word) {
                s.applyPauliRot(wires, inverse, params, word);
            },
            nb::arg("wires"), nb::arg("inverse"), nb::arg("params"),
            nb::arg("word"))
        .def(
            "apply_matrix",
            [](StateT &s, const ArrayC<T> &matrix,
               const std::vector<std::size_t> &wires, bool inverse,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values) {
                s.applyControlledMatrix(matrix.data(), matrix.size(),
                                        controlled_wires, controlled_values,
                                        wires, inverse);
            },
            nb::arg("matrix"), nb::arg("wires"), nb::arg("inverse") = false,
            nb::arg("controlled_wires") = std::vector<std::size_t>{},
            nb::arg("controlled_values") = std::vector<bool>{})
        .def(
            "apply_channel",
            [](StateT &s, const std::vector<ArrayC<T>> &kraus,
               const std::vector<std::size_t> &wires) {
                std::vector<CVec<T>> ks;
                ks.reserve(kraus.size());
                for (const auto &k : kraus) {
                    ks.push_back(toCVec<T>(k));
                }
                s.applyChannel(ks, wires);
            },
            nb::arg("kraus"), nb::arg("wires"),
            "Apply a channel given by its Kraus operators (row-major).")
        .def(
            "apply_noise",
            [](StateT &s, const NM &nm, const NoiseList &noise) {
                for (const auto &[ch, wires] : noise) {
                    s.applyChannel(nm.model.channel(ch), wires);
                }
            },
            nb::arg("model"), nb::arg("noise"),
            "Apply resolved native-noise channels (from lookup_gate / "
            "lookup_readout).")
        .def(
            "probs",
            [](const StateT &s, const std::vector<std::size_t> &wires) {
                return toNumpy<T>(s.probs(wires));
            },
            nb::arg("wires"))
        .def(
            "expval",
            [](const StateT &s, const std::vector<C> &coeffs,
               const std::vector<std::vector<std::size_t>> &wires,
               const std::vector<std::string> &paulis,
               const std::vector<ArrayC<T>> &matrices) {
                return s.expval(makeTerms<T>(coeffs, wires, paulis, matrices));
            },
            nb::arg("coeffs"), nb::arg("wires"), nb::arg("paulis"),
            nb::arg("matrices"),
            "Expectation value of sum_k coeffs[k] * O_k, where O_k is the "
            "Pauli word paulis[k] on wires[k], or matrices[k] if it is "
            "non-empty.")
        .def(
            "var",
            [](const StateT &s, const std::vector<C> &coeffs,
               const std::vector<std::vector<std::size_t>> &wires,
               const std::vector<std::string> &paulis,
               const std::vector<ArrayC<T>> &matrices) {
                return s.var(makeTerms<T>(coeffs, wires, paulis, matrices));
            },
            nb::arg("coeffs"), nb::arg("wires"), nb::arg("paulis"),
            nb::arg("matrices"))
        .def(
            "sample",
            [](const StateT &s, const std::vector<std::size_t> &wires,
               std::size_t shots, std::uint64_t seed) {
                std::mt19937_64 gen(seed);
                auto bits = s.sampleWires(wires, shots, gen);
                std::vector<std::int64_t> out(bits.begin(), bits.end());
                return toNumpy2<std::int64_t>(std::move(out), shots,
                                              wires.size());
            },
            nb::arg("wires"), nb::arg("shots"), nb::arg("seed"),
            "Computational-basis samples of `wires` (shots x len(wires)).")
        .def(
            "collapse",
            [](StateT &s, std::size_t wire, bool branch) {
                s.collapse(wire, branch);
            },
            nb::arg("wire"), nb::arg("branch"))
        .def(
            "measure",
            [](StateT &s, std::size_t wire, int postselect,
               std::uint64_t seed) {
                std::mt19937_64 gen(seed);
                const auto p = s.probs({wire});
                bool outcome = false;
                if (postselect == 0 || postselect == 1) {
                    if (p[postselect] <=
                        std::numeric_limits<T>::epsilon() * 10) {
                        throw nb::value_error(
                            "Probability of postselected state is zero.");
                    }
                    outcome = (postselect == 1);
                } else {
                    std::uniform_real_distribution<T> u(0, 1);
                    outcome = u(gen) >= p[0];
                }
                s.collapse(wire, outcome);
                return outcome;
            },
            nb::arg("wire"), nb::arg("postselect"), nb::arg("seed"),
            "Mid-circuit measurement; postselect is -1, 0 or 1.")
        .def("append_qubit", &StateT::appendQubit)
        .def("trace_out", &StateT::traceOut, nb::arg("wires"))
        .def("reset_qubit", &StateT::resetQubit, nb::arg("wire"))
        .def("copy", [](const StateT &s) { return StateT(s); });
}

} // namespace Pennylane::LightningNoisy::NanoBindings
