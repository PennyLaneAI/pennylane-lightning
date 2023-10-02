#!bin/bash
salloc --nodes 1 -n 4 -c 4 --gpus 4 --qos interactive --time 00:30:00 --constraint gpu --account=m4139

salloc --nodes 4 -n 16 -c 16 --gpus 16 --qos interactive --time 00:15:00 --constraint gpu --account=m4139


salloc --nodes 1 --gpus 4 --qos interactive --time 01:00:00 --constraint gpu --account=m4139
#export MPICH_GPU_SUPPORT_ENABLED=1 
#export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH

module load PrgEnv-gnu cray-mpich cudatoolkit/12.0 craype-accel-nvidia80  gcc cmake
module load evp-patch cray-python

conda create -n gpu-aware-mpich-cuda12 python -y

conda activate gpu-aware-mpich-cuda12

#python tests 
#https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#mpi4py-on-perlmutter
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80  gcc cmake
module load evp-patch cray-python

conda create -n gpu-aware-mpich python -y
conda activate gpu-aware-mpich

MPICC="cc -shared" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

#pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

python setup.py build_ext --define="PLLGPU_ENABLE_MPI=ON" -v
python -m pip install -e . -v
srun -n 2 python -m pytest mpitests
python -m pytest tests

srun -n 2 python mpi_backend.py

srun -n 2 python vqe.py
srun python SEL_benchmark.py &>sel_bench.log&
#cpp tests
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python gcc evp-patch
conda activate gpu-aware-mpi

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export CUQUANTUM_SDK=$HOME/.local/lib/python3.9/site-packages/cuquantum
#cmake -BBuildTests -DPLLGPU_BUILD_TESTS=On  -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment -DCUQUANTUM_SDK=../venv/lib64/python3.9/site-packages/cuquantum
rm -rf BuildTests
cmake -BBuildTests -DPLLGPU_ENABLE_MPI=On -DPLLGPU_BUILD_TESTS=On -DCUQUANTUM_SDK=$CUQUANTUM_SDK
cmake --build BuildTests
srun -n 2 ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner
./BuildTests/pennylane_lightning_gpu/src/tests/runner_gpu


LD_PRELOAD=$PWD/../venv/lib64/python3.9/site-packages/cuquantum/lib/libcustatevec.so.1 srun ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner
cmake -Bbuild -DOPENMPI=1
cmake --build build

ldd ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner | grep libmpi

#pip install -e .
#srun python -m pytest test
#export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH


#####O
#OpenMPI tests on perlmutter
module unload cray-mpich craype-accel-nvidia80 craype  cray-libsci
module load PrgEnv-gnu
module use /global/common/software/m3169/perlmutter/modulefiles
module load openmpi gcc cmake evp-patch cray-python
#module unload cray-mpich craype-accel-nvidia80 craype  cray-libsci

conda create -n gpu-aware-openmpi python -y

conda activate gpu-aware-openmpi

MPICC="/global/common/software/m3169/perlmutter/openmpi/5.0.0rc12-ofi-cuda-22.7_11.7/gnu/bin/mpicc" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

export PATH=/global/common/software/m3169/perlmutter/openmpi/5.0.0rc12-ofi-cuda-22.7_11.7/gnu/bin/:$PATH
export LD_LIBRARY_PATH=/global/common/software/m3169/perlmutter/openmpi/5.0.0rc12-ofi-cuda-22.7_11.7/gnu/lib/:$LD_LIBRARY_PATH
export OMPI_DIR=/global/common/software/m3169/perlmutter/openmpi/5.0.0rc12-ofi-cuda-22.7_11.7/gnu:$OMPI_DIR

export CUQUANTUM_SDK=$HOME/.local/lib/python3.9/site-packages/cuquantum

rm -rf build
python setup.py build_ext --define="PLLGPU_ENABLE_MPI=ON" -v
python -m pip install -e . -v

mpirun --mca smsc none -np 4 python -m pytest mpitests
mpirun -np 4 python -m pytest mpitests

rm -rf BuildTests
cmake -BBuildTests -DSYSTEM_NAME=CrayLinux -DPLLGPU_ENABLE_MPI=On -DPLLGPU_BUILD_TESTS=On -DCUQUANTUM_SDK=$CUQUANTUM_SDK
cmake --build BuildTests

mpirun -np 4 ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner

srun -n 32 python demo_tests/mpi_gpu.py


mpic++ mpi_sparsemv.cpp -o spmv -lmpi_gtl_cuda -L/opt/cray/pe/mpich/8.1.25/gtl/lib
