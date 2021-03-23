// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * @file
 * Defines quantum gates and their actions.
 */
#pragma once

#include <memory>
#include <vector>

#include "StateVector.hpp"
#include "typedefs.hpp"

namespace Pennylane {

    const double SQRT2INV = 0.7071067811865475;
    const CplxType IMAG = CplxType(0, 1);

    class AbstractGate {
    public:
        const int numQubits;
        const size_t length;
    protected:
        AbstractGate(int numQubits);
    public:
        /**
         * @return the matrix representation for the gate as a one-dimensional vector.
         */
        virtual const std::vector<CplxType>& asMatrix() = 0;

        /**
         * Generic matrix-multiplication kernel
         */
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    // Single-qubit gates:

    class SingleQubitGate : public AbstractGate {
    protected:
        SingleQubitGate();
    };

    class XGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static XGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class YGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static YGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class ZGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static ZGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class HadamardGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static HadamardGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class SGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static SGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class TGate : public SingleQubitGate {
    private:
        static const CplxType shift;
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static TGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class RotationXGate : public SingleQubitGate {
    private:
        const CplxType c, js;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static RotationXGate create(const std::vector<double>& parameters);
        RotationXGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
    };

    class RotationYGate : public SingleQubitGate {
    private:
        const CplxType c, s;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static RotationYGate create(const std::vector<double>& parameters);
        RotationYGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
    };

    class RotationZGate : public SingleQubitGate {
    private:
        const CplxType first, second;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static RotationZGate create(const std::vector<double>& parameters);
        RotationZGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class PhaseShiftGate : public SingleQubitGate {
    private:
        const CplxType shift;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static PhaseShiftGate create(const std::vector<double>& parameters);
        PhaseShiftGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class GeneralRotationGate : public SingleQubitGate {
    private:
        const CplxType c, s, r1, r2, r3, r4;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static GeneralRotationGate create(const std::vector<double>& parameters);
        GeneralRotationGate(double phi, double theta, double omega);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
    };

    // Two-qubit gates

    class TwoQubitGate : public AbstractGate {
    protected:
        TwoQubitGate();
    };

    class CNOTGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CNOTGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class SWAPGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static SWAPGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CZGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CZGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CRotationXGate : public TwoQubitGate {
    private:
        const CplxType c, js;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CRotationXGate create(const std::vector<double>& parameters);
        CRotationXGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CRotationYGate : public TwoQubitGate {
    private:
        const CplxType c, s;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CRotationYGate create(const std::vector<double>& parameters);
        CRotationYGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CRotationZGate : public TwoQubitGate {
    private:
        const CplxType first, second;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CRotationZGate create(const std::vector<double>& parameters);
        CRotationZGate(double rotationAngle);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CGeneralRotationGate : public TwoQubitGate {
    private:
        const CplxType c, s, r1, r2, r3, r4;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CGeneralRotationGate create(const std::vector<double>& parameters);
        CGeneralRotationGate(double phi, double theta, double omega);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    // Three-qubit gates

    class ThreeQubitGate : public AbstractGate {
    protected:
        ThreeQubitGate();
    };

    class ToffoliGate : public ThreeQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static ToffoliGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    class CSWAPGate : public ThreeQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CSWAPGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    // General gates
    class QubitUnitary : public AbstractGate {
    private:
        const std::vector<CplxType> matrix;
    public:
        QubitUnitary(const int numQubits, std::vector<CplxType> const &mx);
        static const std::string label;
        static QubitUnitary create(const std::vector<double>& parameters);
        static QubitUnitary create(const unsigned int qubit_num, const std::vector<CplxType>& matrix) {         Pennylane::QubitUnitary(qubit_num, matrix); };
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
    };

    /**
     * Produces the requested gate, defined by a label and the list of parameters
     *
     * @param label unique string corresponding to a gate type
     * @param parameters defines the gate parameterisation (may be zero-length for some gates)
     * @return the gate wrapped in std::unique_ptr
     * @throws std::invalid_argument thrown if the gate type is not defined, or if the number of parameters to the gate is incorrect
     */
    std::unique_ptr<AbstractGate> constructGate(const std::string& label, const std::vector<double>& parameters);
    std::unique_ptr<AbstractGate> constructGate(const std::vector<CplxType>& matrix);

}
