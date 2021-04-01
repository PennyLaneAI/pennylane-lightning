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
        virtual const std::vector<CplxType>& asTargetMatrix() = 0;
        virtual const std::vector<unsigned int>& getAllWires() = 0;
        virtual const std::vector<unsigned int>& getControlWires() = 0;
        virtual const std::vector<unsigned int>& getTargetWires() = 0;


        /**
         * Generic matrix-multiplication kernel
         */
        virtual void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);

        /**
         * Kernel for applying the generator of an operation
         */
        virtual void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);

        /**
         * Scaling factor applied to the generator operation
         */
        static const double generatorScalingFactor;
    };

    // Single-qubit gates:

    class SingleQubitGate : public AbstractGate {
    protected:
        SingleQubitGate();
    };

    class XGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static XGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        XGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class YGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static YGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        YGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class ZGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static ZGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        ZGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class HadamardGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static HadamardGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        HadamardGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class SGate : public SingleQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static SGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        SGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class TGate : public SingleQubitGate {
    private:
        static const CplxType shift;
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static TGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        TGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class RotationXGate : public SingleQubitGate {
    private:
        const CplxType c, js;
        const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static RotationXGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        RotationXGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class RotationYGate : public SingleQubitGate {
    private:
        const CplxType c, s;
        const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static RotationYGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        RotationYGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class RotationZGate : public SingleQubitGate {
    private:
        const CplxType first, second;
        const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static RotationZGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        RotationZGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
    };

    class PhaseShiftGate : public SingleQubitGate {
    private:
        const CplxType shift;
        const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static PhaseShiftGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        PhaseShiftGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
    };

    class GeneralRotationGate : public SingleQubitGate {
    private:
        const CplxType c, s, r1, r2, r3, r4;
        const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static GeneralRotationGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        GeneralRotationGate(double phi, double theta, double omega, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    // Two-qubit gates

    class TwoQubitGate : public AbstractGate {
    protected:
        TwoQubitGate();
    };

    class CNOTGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        static const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CNOTGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CNOTGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class SWAPGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static SWAPGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        SWAPGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class CZGate : public TwoQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        static const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CZGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CZGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class CRotationXGate : public TwoQubitGate {
    private:
        const CplxType c, js;
        const std::vector<CplxType> matrix;
        const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CRotationXGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CRotationXGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
    };

    class CRotationYGate : public TwoQubitGate {
    private:
        const CplxType c, s;
        const std::vector<CplxType> matrix;
        const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CRotationYGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CRotationYGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
    };

    class CRotationZGate : public TwoQubitGate {
    private:
        const CplxType first, second;
        const std::vector<CplxType> matrix;
        const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CRotationZGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CRotationZGate(double rotationAngle, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
        void applyGenerator(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices);
        static const double generatorScalingFactor;
    };

    class CGeneralRotationGate : public TwoQubitGate {
    private:
        const CplxType c, s, r1, r2, r3, r4;
        const std::vector<CplxType> matrix;
        const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CGeneralRotationGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CGeneralRotationGate(double phi, double theta, double omega, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    // Three-qubit gates

    class ThreeQubitGate : public AbstractGate {
    protected:
        ThreeQubitGate();
    };

    class ToffoliGate : public ThreeQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static ToffoliGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        ToffoliGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return matrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    class CSWAPGate : public ThreeQubitGate {
    private:
        static const std::vector<CplxType> matrix;
        static const std::vector<CplxType> targetMatrix;
        const std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;

    public:
        static const std::string label;
        static CSWAPGate create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        CSWAPGate(const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return matrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        const std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
        void applyKernel(const StateVector& state, const std::vector<size_t>& indices, const std::vector<size_t>& externalIndices, bool inverse);
    };

    // General gates
    class QubitUnitary : public AbstractGate {
    private:
        const std::vector<CplxType> matrix;
        const std::vector<CplxType> targetMatrix;
        std::vector<unsigned int> allWires;
        const std::vector<unsigned int> controlWires;
        const std::vector<unsigned int> targetWires;
    public:
        QubitUnitary(const int numQubits, std::vector<CplxType> const &mx, const std::vector<unsigned int>& controlWires, const std::vector<unsigned int>& targetWires);
        static const std::string label;
        static QubitUnitary create(const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
        inline const std::vector<CplxType>& asMatrix() {
            return targetMatrix;
        }
        inline const std::vector<CplxType>& asTargetMatrix(){ return targetMatrix; }
        std::vector<unsigned int>& getAllWires() { return allWires; }
        const std::vector<unsigned int>& getControlWires() { return controlWires; }
        const std::vector<unsigned int>& getTargetWires() { return targetWires; }
    };

    /**
     * Produces the requested gate, defined by a label and the list of parameters
     *
     * @param label unique string corresponding to a gate type
     * @param parameters defines the gate parameterisation (may be zero-length for some gates)
     * @return the gate wrapped in std::unique_ptr
     * @throws std::invalid_argument thrown if the gate type is not defined, or if the number of parameters to the gate is incorrect
     */
    std::unique_ptr<AbstractGate> constructGate(const std::string& label, const std::vector<double>& parameters, const std::vector<unsigned int>& wires);
    std::unique_ptr<AbstractGate> constructGate(const std::vector<CplxType>& matrix, const std::vector<unsigned int>& controlWires, const std::vector<unsigned int>& targetWires);

}
