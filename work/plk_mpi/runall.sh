export OMP_PROC_BIND=true
for nq in {20..29}; do
    for nproc in 32 16 8 4 2 1; do
        OMP_NUM_THREADS=1 mpiexec -n $nproc --map-by ppr:1:core:pe=1 python bench.py mpi $nq
        # OMP_NUM_THREADS=$nproc python bench.py openmp $nq
        # OMP_NUM_THREADS=$((32 / nproc)) mpiexec -n $nproc --map-by ppr:$nproc:node:pe=$((32 / nproc)) python bench.py mpi_openmp $nq
        # mpiexec -n $nproc --map-by ppr:$nproc:node:pe=1 python bench.py cuda $nq
        # python bench.py cuda $nq
    done
done