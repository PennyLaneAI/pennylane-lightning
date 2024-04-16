#include "DevTag.hpp"
#include "MPS_cuDevice.hpp"
#include <complex>
#include <iostream>
#include <vector>

int main() {
    using namespace Pennylane::LightningTensor;
    size_t numQubits = 3;
    size_t maxExtent = 2;
    std::vector<size_t> qubitDims(numQubits, 2);
    std::cout << "Quantum circuit: " << numQubits << " qubits\n";
    Pennylane::LightningGPU::DevTag<int> dev_tag(0, 0);

    MPS_cuDevice<double> mps(numQubits, maxExtent, qubitDims, dev_tag);

    size_t index = 7;
    mps.setBasisState(index);

    std::string opName = "PauliX";
    std::vector<size_t> wires = {2};
    mps.applyOperation(opName, wires);

    auto finalState = mps.getStateVector();

    for (auto &element : finalState) {
        std::cout << element << std::endl;
    }

    return 0;
}