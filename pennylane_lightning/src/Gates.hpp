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

#include <vector>

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
    };

    class TGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static TGate create(const std::vector<double>& parameters);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
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
    };

    class GeneralRotationGate : public SingleQubitGate {
    private:
        const CplxType c, s;
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
    };

    class CRotationYGate : public SingleQubitGate {
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
    };

    class CRotationZGate : public SingleQubitGate {
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
    };

    class CGeneralRotationGate : public SingleQubitGate {
    private:
        const CplxType c, s;
        const std::vector<CplxType> matrix;
    public:
        static const std::string label;
        static CGeneralRotationGate create(const std::vector<double>& parameters);
        CGeneralRotationGate(double phi, double theta, double omega);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
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
    };

}
