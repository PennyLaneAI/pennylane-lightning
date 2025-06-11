from .gpu_binding_example_ext import DeviceTensor, add_tensors

# Define what `from gpu_binding_example import *` will import
__all__ = [
    "DeviceTensor",
    "add_tensors",
]
