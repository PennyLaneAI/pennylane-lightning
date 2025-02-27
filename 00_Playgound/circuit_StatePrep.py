import pennylane as qml
import numpy as np

def example_circuit(device_name:str, wires:int, state:np.ndarray, state_wires:list):
    
    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def circuit(state=state, state_wires=state_wires):
        qml.StatePrep(state, wires=state_wires)
        # return qml.state()
        return qml.expval(qml.PauliZ(0))

    return circuit

def initial_state(wires:int, is_random:bool=False):
    
    if is_random:
        state = np.random.rand(2**wires) + 1j*np.random.rand(2**wires)
        state = state/np.linalg.norm(state)
    else:
        state = np.zeros(2**wires) + 0j
        state[0] = 1 + 0j
        
    return state

if __name__ == "__main__":
    
    wires = 6
    device_name = 'lightning.qubit'
    
    
    state_wires = [i for i in range(0, wires)]
    state = initial_state(len(state_wires), is_random=True)
    
    print(f"Creating a circuit")
    print(f"{device_name=}")
    print(f"{wires=}, {state_wires=}")
    
    circuit = example_circuit(device_name, wires, state, state_wires)
    
    print(f"Running the circuit")
    print(f"result = {circuit()}")