#include "tensor.hpp" // contains custom DeviceTensor class

#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <vector>
#include <numeric> // For std::iota

namespace nb = nanobind;

// Define a view datatype based on the specific ndarray type we want to create.
using DeviceView =
    nb::ndarray<float, nb::shape<-1>, nb::c_contig, nb::device::cuda>;

using NumpyArray = nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig>;

// Forward-declare the CUDA kernel launcher from kernels.cu
void launch_add_vectors(float *out, const float *a, const float *b, int n);

// The C++ function that performs the GPU operation
void add_tensors(DeviceTensor &out, const DeviceTensor &a,
                 const DeviceTensor &b) {
    // Basic validation
    if (a.size() != b.size() || a.size() != out.size()) {
        throw std::runtime_error("Tensor shapes must match.");
    }

    // Get the raw device pointers from our wrapper objects
    launch_add_vectors(out.data(), a.data(), b.data(), a.size());
}

DeviceTensor *device_tensor_from_array(NumpyArray &arr) {
    std::cout << "Copying array of size " << arr.size() << std::endl;

    float *data = arr.data();
    std::cout << "Data pointer: " << data << std::endl;
    data[0] = 42.0f; // Example operation to show we can access data

    DeviceTensor *tens = new DeviceTensor(arr.size());

    cudaMemcpy(tens->data(), data, arr.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return tens;
}

// Returns a Python view to the on-device tensor object
DeviceView device_tensor_gpu_view(nb::object self) {
    // Get a reference to the C++ instance from the Python 'self' object. Needed
    // as a handle/owner
    DeviceTensor *t = nb::inst_ptr<DeviceTensor>(self);

    // Get the device ID (assuming device 0 for this example)
    int device_id = 0;
    // cudaGetDevice(&device_id);

    size_t ndim = 1;
    size_t shape[] = {t->size()};

    return DeviceView(t->data(), ndim, shape, self);
}

std::vector<float> generate_sequential_data(size_t count) {
    std::vector<float> data(count);
    std::iota(data.begin(), data.end(), 0.0f); // Fill with 0, 1, 2, ...
    return data; // Return by value
}

// The module name must match what's in CMakeLists.txt and setup.py
NB_MODULE(gpu_binding_example_ext, m) {
    m.doc() = "A GPU tensor management example with nanobind";

    nb::class_<DeviceTensor>(m, "DeviceTensor")
        .def(nb::init<size_t>())
        // Expose the shape as a read-only property
        .def_prop_ro("size", [](const DeviceTensor &t) { return t.size(); })

        .def("gpu_view", &device_tensor_gpu_view,
             "Returns a zero-copy view of the tensor on the GPU (DLPack "
             "compatible).")
        .def("print_data", &DeviceTensor::print_data,
             "Prints the data currently stored within the GPU.")

        // Needed for Python to be able to print this object
        .def("__repr__", &DeviceTensor::repr,
             "Returns a string representation of the DeviceTensor.");

    // Bind the function that operates on our DeviceTensors
    m.def("add_tensors", &add_tensors,
          "Adds two DeviceTensors using a CUDA kernel.", nb::arg("out"),
          nb::arg("a"), nb::arg("b"));

    // Bind factory function to create a DeviceTensor from a NumPy array
    m.def("device_tensor_from_array", &device_tensor_from_array,
          "Create a DeviceTensor from an existing NumPy array", nb::arg("arr"),
          nb::rv_policy::take_ownership);

    m.def("generate_sequential_data", [](size_t count) {
            std::vector<float> v = generate_sequential_data(count);

            std::vector<float> *v2 = new std::vector<float>(std::move(v));
            float* data_ptr = v2->data();

            // float* v2 = new float[count];
            // float* data_ptr = v2;

            size_t shape[] = {count};

            auto capsule = nb::capsule(
                v2, [](void *ptr) noexcept {
                    delete static_cast<std::vector<float> *>(ptr); 
                    // delete[] static_cast<float *>(ptr); 
                });

            return NumpyArray(
                data_ptr, 1, shape, capsule);
        },
          "Generates sequential float data and returns it as a NumPy array (zero-copy).");
}
