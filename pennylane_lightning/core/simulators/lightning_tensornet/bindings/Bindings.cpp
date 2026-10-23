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
 * @file Bindings.cpp
 * Python module `lightning_tensornet_ops`: Kokkos matrix-product-state
 * (pure) / matrix-product-density-operator (mixed) simulator.
 */
#include <complex>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>

#include "MPSKokkos.hpp"
#include "MPSLinAlg.hpp"
#include "NoisyBindings.hpp"

namespace nb = nanobind;
using namespace Pennylane::LightningNoisy::NanoBindings;
using Pennylane::LightningTensornet::MPSKokkos;

namespace {

template <class T> void registerMPS(nb::module_ &m, const std::string &suffix) {
    using MPS = MPSKokkos<T>;
    using C = std::complex<T>;
    registerNoiseModel<MPS>(m, "NativeNoiseModelC" + suffix);

    auto cls = nb::class_<MPS>(m, ("MPSC" + suffix).c_str());
    cls.def(
           "__init__",
           [](MPS *self, std::size_t num_qubits, long long max_bond_dim,
              double cutoff, const std::string &cutoff_mode, bool mixed,
              const nb::dict &kokkos_args) {
               if (max_bond_dim <= 0) {
                   throw nb::value_error("max_bond_dim must be positive.");
               }
               if (!(cutoff >= 0)) {
                   throw nb::value_error("cutoff must be non-negative.");
               }
               if (cutoff_mode != "abs" && cutoff_mode != "rel") {
                   throw nb::value_error("cutoff_mode must be 'abs' or 'rel'.");
               }
               new (self)
                   MPS(num_qubits, static_cast<std::size_t>(max_bond_dim),
                       static_cast<T>(cutoff), cutoff_mode, mixed,
                       kokkosSettings(kokkos_args));
           },
           nb::arg("num_qubits"), nb::arg("max_bond_dim") = 128,
           nb::arg("cutoff") = 0.0, nb::arg("cutoff_mode") = "abs",
           nb::arg("mixed") = false, nb::arg("kokkos_args") = nb::dict())
        .def_prop_ro("max_bond_dim", &MPS::getMaxBondDim)
        .def_prop_ro("cutoff", &MPS::getCutoff)
        .def_prop_ro("cutoff_mode", &MPS::getCutoffMode)
        .def("is_mixed", &MPS::isMixed, "True in density-operator (MPDO) mode.")
        .def("to_mixed", &MPS::toMixed,
             "Switch to density-operator mode (exact; bonds are squared).")
        .def("bond_dims", &MPS::bondDims,
             "Bond dimensions D_0..D_N (D_0 = D_N = 1).")
        .def("truncation_error", &MPS::truncationError,
             "Sum over all truncating SVDs of the discarded fraction of the "
             "squared singular values.")
        .def(
            "state_vector",
            [](const MPS &s) {
                if (s.isMixed()) {
                    throw nb::value_error(
                        "The state is mixed (density-operator mode); the "
                        "state vector is not defined.");
                }
                return toNumpy<C>(s.stateVector());
            },
            "Dense state vector (pure mode only; wire 0 is the most "
            "significant bit).")
        .def(
            "get_density_matrix",
            [](const MPS &s) {
                const std::size_t d = std::size_t{1} << s.getNumQubits();
                return toNumpy2<C>(s.getDensityMatrix(), d, d);
            },
            "Full density matrix (2^N x 2^N, normalized).")
        .def(
            "reduced_density_matrix",
            [](const MPS &s, const std::vector<std::size_t> &wires) {
                const std::size_t d = std::size_t{1} << wires.size();
                return toNumpy2<C>(s.reducedDensityMatrix(wires), d, d);
            },
            nb::arg("wires"))
        .def("purity", &MPS::purity)
        .def("trace", &MPS::trace)
        .def(
            "site_tensors",
            [](const MPS &s) {
                std::vector<nb::ndarray<nb::numpy, C, nb::ndim<1>>> out;
                for (auto &t : s.siteTensors()) {
                    out.push_back(toNumpy<C>(std::move(t)));
                }
                return out;
            },
            "Site tensors (row-major D_i x d x D_{i+1}; d = 2 pure, 4 mixed).");
    registerCommonState<MPS>(cls);
}

} // namespace

NB_MODULE(lightning_tensornet_ops, m) {
#ifdef NDEBUG
    nb::set_leak_warnings(false);
#endif
    m.doc() = "PennyLane Lightning Tensornet: Kokkos MPS / MPDO simulator.";
    registerKokkosInfo(m);
    m.def("backend_info", []() {
        std::map<std::string, std::string> info;
        info["NAME"] = "lightning.tensornet";
        info["EXECUTION_SPACE"] = Kokkos::DefaultExecutionSpace::name();
        info["SVD"] = Pennylane::LightningTensornet::svdUsesLapack() ? "lapack"
                                                                     : "jacobi";
        return info;
    });
    registerMPS<float>(m, "64");
    registerMPS<double>(m, "128");
}
