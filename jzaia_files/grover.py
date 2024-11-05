import numpy as np
import pennylane as qml

# Define oracles
ORACLE1_QUBITS   = 6
ORACLE1_EXPECTED = [1, 1, 0, 1, 0]

ORACLE2_QUBITS   = 10
ORACLE2_EXPECTED = [1, 0, 1, 0, 1, 0, 1, 0, 1]

ORACLE3_QUBITS   = 17
ORACLE3_EXPECTED = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]

ORACLES = [(ORACLE1_QUBITS, ORACLE1_EXPECTED),
           (ORACLE2_QUBITS, ORACLE2_EXPECTED),
           (ORACLE3_QUBITS, ORACLE3_EXPECTED)]


def grovers_setup(num_qubits: int):
    '''
    Setup up a circuit for iterations of Grover's search

    Places a circuit into uniform superposition. Additionally, places the
    ancilla qubit for the oracle in the |-> state such that it can
    apply phase kickback.

    :param num_qubits: The number of qubits in the circuit
    '''
    qml.X(num_qubits-1)
    for i in range(num_qubits):
        qml.Hadamard(wires=i)

def grovers_mirror(num_qubits):
    '''
    Apply the "Grover's Mirror" reflection to the active circuit

    Performs a reflection across the vector which represents the uniform
    superposition. This is used for amplitude-amplification for Grover's
    algorithm.

    :param num_qubits: The number of qubits in the circuit
    '''
    for i in range(num_qubits-1):
        qml.Hadamard(wires=i)

    qml.MultiControlledX(wires=range(num_qubits),
                         control_values=[False]*(num_qubits-1))

    for i in range(num_qubits-1):
        qml.Hadamard(wires=i)

def run_grovers(oracle, num_qubits):
    '''
    Overall function for running Grover's algorithm on a chosen oracle

    Run Grover's algorithm from start to finish. Prepares a state, and
    repeats state selection and amplitude-amplification for sqrt(N) iterations
    (where N = 2^(# of non-ancilla qubits)). This implementation assumes that
    the oracle always picks precisly 1 state (rather than an arbitrary number).

    :param oracle: A black-box function that acts on a created statevector
    :param num_qubits: The number of qubits in the circuit the oracle acts
           on (includes the ancilla)
    '''
    grovers_setup(num_qubits)

    reps = int(np.sqrt(2**(num_qubits-1)))
    for _ in range(reps):
        oracle()
        grovers_mirror(num_qubits)

    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits-1)]

def run_experiment(oracle, num_qubits) -> None:
    '''
    Run Grover's algorithm and evaluates results

    Run Grover's algorithm from start to finish, and finds the expected
    measurement outcome.

    :param oracle: A black-box function that acts on a created statevector
    :param num_qubits: The number of qubits in the circuit the oracle acts
           on (includes the ancilla)
    '''
    dev = qml.device('lightning.qubit', wires=num_qubits)


    circ = qml.QNode(run_grovers, dev)

    expvals = circ(oracle, num_qubits)
    results = [int(val.numpy() < 0) for val in expvals]

    print(results)

def gen_oracle(i):
    '''
    Create an oracle function which selects the state given by the global const

    :param i: The index of the globally defined pair of constants to use
    '''
    num_qubits   = ORACLES[i][0]
    control_vals = ORACLES[i][1]
    def oracle():
        qml.MultiControlledX(wires=range(num_qubits),
                             control_values=control_vals)
    return (oracle, num_qubits)


if __name__ == '__main__':
    # Run all experiments
    for i in range(len(ORACLES)):
        print('Expecting:', ORACLES[i][1])
        print('Got:')
        run_experiment(*gen_oracle(i))
        print()
