#!bin/bash
salloc --nodes 1 -n 4 -c 4 --gpus 4 --qos interactive --time 00:50:00 --constraint gpu --account=m4139

module load PrgEnv-gnu cray-mpich cudatoolkit/12 craype-accel-nvidia80  gcc cmake
module load evp-patch

conda activate gpu-aware-mpich


#LightningGPU cpp layer installation
cmake -B build -DCUQUANTUM_SDK=$HOME/.conda/envs/py311-cu12/lib/python3.11/site-packages/cuquantum -DBUILD_TESTS=ON -DPL_BACKEND=lightning_gpu

cmake -B build -DCMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0:/opt/cray/pe/mpich/8.1.25/gtl/lib/ -DENABLE_MPI=ON -DCUQUANTUM_SDK=$HOME/.conda/envs/py311-cu12/lib/python3.11/site-packages/cuquantum -DBUILD_TESTS=ON -DPL_BACKEND=lightning_gpu
cmake --build build

#LightningQubits


#python installation

CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/math_libs/12.0 -DENABLE_MPI=ON" PL_BACKEND="lightning_gpu" python -m pip install -e .
export PL_DEVICE=lightning_gpu

conda deactivate
module load PrgEnv-nvidia cray-mpich cudatoolkit/12 craype-accel-nvidia80 python/3.11 gcc/11.2 cmake
module load evp-patch
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:/opt/cray/pe/mpich/8.1.25/gtl/lib/:$LD_LIBRARY_PATH

conda create -n py311-cu12 python=3.11 -y
conda activate py311-cu12
conda deactivate py311-cu12

MPICC="cc -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py



#cu11 support
conda deactivate
module load PrgEnv-nvidia cray-mpich cudatoolkit/11.7 craype-accel-nvidia80 python/3.11 gcc/11.2 cmake
module load evp-patch
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:/opt/cray/pe/mpich/8.1.25/gtl/lib/:$LD_LIBRARY_PATH

conda create -n py311-cu11 python=3.11 -y
MPICC="cc -shared" CC=nvc pip install --force --no-cache-dir --no-binary=mpi4py mpi4py

pip install cuquantum-cu11

cmake -B build -DCUQUANTUM_SDK=$HOME/.conda/envs/py311-cu11/lib/python3.11/site-packages/cuquantum -DBUILD_TESTS=ON -DPL_BACKEND=lightning_gpu
cmake --build build

PL_BACKEND="lightning_gpu" python -m pip install -e .
python -m pytest tests
