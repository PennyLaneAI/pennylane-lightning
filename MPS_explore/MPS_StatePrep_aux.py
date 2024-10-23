import numpy as np
import pennylane as qml



# Function to define the bond dim for MPS on base to the number of qubits.
def setBondDims(numQubits, maxBondDim):
    
    log_maxBondDim = np.log2(maxBondDim)
    localBondDims = np.ones(numQubits -1) * maxBondDim
    
    for i, val in enumerate(localBondDims):
        bondDim = min(i+1, numQubits - i - 1)
        if bondDim <= log_maxBondDim:
            localBondDims[i] = 2**bondDim
    
    return localBondDims
    
def setSitesModes(numQubits):
    localSitesModes = []
    
    for i in range(numQubits):
        if i == 0:
            localSite = [i, i+numQubits, -1]
        elif i == numQubits - 1:
            localSite = [i+numQubits-1, i, -1]
        else:
            localSite = [i + numQubits - 1, i , i + numQubits]
        
        localSitesModes.append(localSite)
    
    localSitesModes = np.array(localSitesModes)
    
    return localSitesModes

def setSitesExtents(numQubits, bondDims):
    qubitDims = np.ones(numQubits,dtype=int) * 2
    localSiteExtents = []
    for i in range(numQubits):
        if i == 0:
            localSite = [qubitDims[i], bondDims[i], -1]
        elif i == numQubits - 1:
            localSite = [bondDims[i-1], qubitDims[i], -1]
        else:
            localSite = [bondDims[i-1], qubitDims[i], bondDims[i]]
        
        localSiteExtents.append(localSite)
    
    localSiteExtents = np.array(localSiteExtents).astype(int)
    
    return localSiteExtents

# Function to print all the array for set the MPS bond dim.
def build_MPS(numQubits, maxBondDim):
    print('Conditions')
    print('NumQubit:', numQubits, '| MaxBondDim:',maxBondDim)


    print('Function setBondDims')
    bondDims = setBondDims(numQubits, maxBondDim)
    
    print('len:', bondDims.shape)
    print(bondDims.astype(int))
    
    print('-'*100)
    
    print('Function setSitesModes')
    sitesModes = setSitesModes(numQubits)
    
    print('shape',sitesModes.shape)
    print(sitesModes.T)

    print('-'*100)
    
    print('Function setSiteExtends')
    siteExtents = setSitesExtents(numQubits, bondDims)

    print("Sites Extents")
    print('shape',siteExtents.shape)
    print(siteExtents.T)

    print('-'*100)

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# Create a random MPS with the correct dimensions 
def create_custom_MPS(numQubits):
    
    maxBondDim = 128
    
    bondDims = setBondDims(numQubits, maxBondDim)
    sitesExtents = setSitesExtents(numQubits, bondDims)
    
    MPS_example = []
    
    for T_shape in sitesExtents:
        
        if T_shape[-1] == -1:
            T_shape = T_shape[:-1]
            
        MPS_site = np.random.rand(*(tuple(T_shape)))
        
        MPS_example.append(MPS_site)
    
    return MPS_example

# Dummy circuit
def LTensor_circuit(wires, custom_MPS):
    
    # It is necessary to create an fake state to avoid complete from the preprocess 
    state_one = np.zeros(2**(wires-1), dtype=float)
    state_one = state_one + state_one * 0j
    state_one[0] = 1.0 + 0j
    
    dev = qml.device("lightning.tensor", wires=wires, cutoff=1e-7, max_bond_dim=128)
    
    dev_wires = dev.wires.tolist()

    @qml.qnode(dev)
    def circuit(set_state=state_one):
        
        qml.StatePrep(set_state, wires=dev_wires[1:])
        
        return qml.state()
        
    result = circuit()
    
    # print(f"Circuit result: \n{result}")

# Dummy circuit
def LTensor_circuit_MPS(wires, custom_MPS):
    
    dev = qml.device("lightning.tensor", wires=wires, cutoff=1e-7, max_bond_dim=128)

    dev_wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def circuit(custom_MPS=custom_MPS):
        
        qml.MPSPrep(custom_MPS, wires=dev_wires[1:])
        
        return qml.state()
        
    result = circuit()
    
    # print(f"Circuit result: \n{result}")


if __name__ == '__main__':
    
    wires = 9
    maxBondDim = 128
    
    build_MPS(wires, maxBondDim)

    # ----------------------------------------------------------------------
    print('-'*100)
    print("Lightning Tensor Base")

    LTensor_circuit(wires,[]) # Empty custom_MPS
    
    # ----------------------------------------------------------------------
    print('-'*100)
    print("Lightning Tensor Custom MPS")
    
    print('Custom MPS shape')
    MPS_example = create_custom_MPS(wires)
    [print(f'{site.shape}') for site in MPS_example]

    LTensor_circuit_MPS(wires, MPS_example) # Passing a custom_MPS
    
    
    