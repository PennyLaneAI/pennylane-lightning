import numpy as np
import pennylane as qml


# Function to define the bond dim for MPS on base to the number of qubits.
def setBondDims(numQubits, maxBondDim):
    
    log_maxBondDim = np.log2(maxBondDim)
    limit_dimension = 2 ** int(log_maxBondDim)
    localBondDims = np.ones(numQubits -1) * limit_dimension
    
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

# Function to print an MPS.
def print_mps(mps, full=False, name=None):
    """Print the MPOs."""
    
    max_sites_length = 10
    
    print("-"*100)
    if name is not None:
        print("MPS name:", name)
    
    max_bond_dim = 0
    for i, site in enumerate(mps):
        max_bond_dim = max(max_bond_dim, site.shape[-1])
    
    print("Sites:", len(mps), "| Max bond dim:", max_bond_dim)
    
    max_rows = len(mps)//max_sites_length
    if len(mps) % max_sites_length != 0:
        max_rows += 1
    
    for rows in range(max_rows):
        for i in range(rows*max_sites_length, (rows+1)*max_sites_length):
            print(f"| {'site '+str(i):^12}",end="")
        
        print()
        for i, site in enumerate(mps[rows*max_sites_length:(rows+1)*max_sites_length]):
            print(f"| {str(site.shape):^12}", end="")
        print()
        print()
    
    if full:
        print("~"*100)
        for i, site in enumerate(mps):
            # print("Site", i, ":\n", site)
            print("Site", i, " | Shape:", site.shape)            
            print(site)
            print("~"*100)
    print("-"*100)

# Create a random MPS with the custom dimensions
def create_MPS_with_custom_bondDims(numQubits,bondDims):
    
    sitesExtents = setSitesExtents(numQubits, bondDims)
    
    MPS_example = []
    
    for T_shape in sitesExtents:
        
        if T_shape[-1] == -1:
            T_shape = T_shape[:-1]
            
        MPS_site = np.random.rand(*(tuple(T_shape)))
        
        MPS_example.append(MPS_site)
    
    sitesExtents = sitesExtents.tolist()

    sitesExtents[0] = sitesExtents[0][:-1]
    sitesExtents[-1] = sitesExtents[-1][:-1]
    
    # flatten the list
    sitesExtents = [item for sublist in sitesExtents for item in sublist]
    
    return MPS_example, sitesExtents

if __name__ == "__main__":
    
    wires = 4
    bond_dims = [5,7,5]
    
    MPS_example, siteExtents = create_MPS_with_custom_bondDims(wires, bond_dims)
    
    method = "mps"
    
    dev = qml.device("lightning.tensor", wires=wires, method=method, cutoff=1e-7, bond_dim=bond_dims)

    dev_wires = dev.wires.tolist()
    
    @qml.qnode(dev)
    def circuit(custom_MPS=MPS_example):
        
        qml.MPSPrep(mps=custom_MPS, wires=dev_wires)
        
        return qml.expval(qml.PauliZ(0))
        
    print("result",circuit())

