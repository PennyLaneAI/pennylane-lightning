import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension):
        # The output directory for the compiled .so file
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DCMAKE_BUILD_TYPE=Debug"
        ]

        # Build directory
        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- Running CMake Configure in {build_dir} ---")
        subprocess.run(
            ["cmake", str(Path(__file__).parent.absolute())] + cmake_args,
            cwd=build_dir,
            check=True
        )

        print("--- Running CMake Build ---")
        subprocess.run(
            ["cmake", "--build", "."],
            cwd=build_dir,
            check=True
        )

# This name must match the name defined in the NB_MODULE macro in bindings.cpp
cmake_extension = Extension(
    "gpu_binding_example.gpu_binding_example_ext",
    sources=[]
)

setup(
    ext_modules=[cmake_extension],

    cmdclass={"build_ext": CMakeBuild},
    packages=["gpu_binding_example"],

    zip_safe=False,
)