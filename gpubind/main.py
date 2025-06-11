import numpy as np
import gc
import gpu_binding_example as cuda_ext
import cupy as cp # Import CuPy

def main():
    size = 10
    
    in_tensor1 = cuda_ext.DeviceTensor(size)
    in_tensor2 = cuda_ext.DeviceTensor(size)
    out_tensor = cuda_ext.DeviceTensor(size)
    
    # Get a zero-copy GPU view of the data
    gpu_view = in_tensor1.gpu_view()
    print(f"GPU View object type: {type(gpu_view)}")

    # Get a handle to the array via CuPy
    cupy_array = cp.from_dlpack(gpu_view)

    # The goal: prove that they share the same memory
    
    print("Modifying data on GPU via CuPy view (cupy_array[0] = 123.45)...")
    cupy_array[0] = 123.45

    # Copy the data back to the CPU using the *original* C++ object.
    #    If they share memory, the change we just made should be reflected.
    print("Copying data to CPU from original C++ object...")

    cpu_numpy_array = cp.from_dlpack(in_tensor1.gpu_view()).get()
    print(f"CPU NumPy array: {cpu_numpy_array}")

    print(f"\nValue at index 0 on CPU: {cpu_numpy_array[0]}")
    if np.isclose(cpu_numpy_array[0], 123.45):
        print("SUCCESS: The change is reflected. Memory is shared on the GPU!")
    else:
        print("FAILURE: Memory is not shared.")


    # Try executing a CUDA kernel to add two tensors
    cuda_ext.add_tensors(out_tensor, in_tensor1, in_tensor2)
    cpu_numpy_array = cp.from_dlpack(out_tensor.gpu_view()).get()

    # Print the result from both Python and C++, check for match
    print(f"Out tensor: {cpu_numpy_array}")
    out_tensor.print_data()


    # Next goal: prove that memory is kept alive.
    #    Even if we delete the original handle, the CuPy array's view
    #    should keep the C++ memory alive.
    print("\n--- Lifetime Management Demonstration ---")
    print("Deleting original python handles...")
    del in_tensor1
    del in_tensor2
    del out_tensor
    gc.collect() # Ask garbage collector to run

    # At this point, tensor 2 and the out tensor should be deallocated

    # The "Deallocated" message from tensor 1 should NOT appear yet.
    # Now, we modify the CuPy array again. If the memory was freed, this would crash.
    print("Modifying data again via CuPy view after deleting original handle...")
    cupy_array[1] = 678.9
    cp.cuda.runtime.deviceSynchronize() # ensure operation is complete
    print("Modification successful. Memory was correctly kept alive by the view.")

    # Now, when cupy_array goes out of scope, the handle is dropped and 
    # the C++ destructor will finally run.
    print("\nScript finished. C++ object will be deallocated now.")


if __name__ == "__main__":
    main()