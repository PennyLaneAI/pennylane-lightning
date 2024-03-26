# %%
import numpy as np


# %%
def read_dat(filename):
    with open(filename, "r") as file:
        text = file.read()

    text = text.split("\n")
    nprocs, nthreads, nqbits, times = [], [], [], []
    for line in text:
        if not line:
            break
        nm, nt, nq, t = line.split()
        nprocs.append(int(nm))
        nthreads.append(int(nt))
        nqbits.append(int(nq))
        times.append(float(t))

    nprocs = np.array(nprocs, dtype=int)
    nthreads = np.array(nthreads, dtype=int)
    nqbits = np.array(nqbits, dtype=int)
    times = np.array(times, dtype=float)
    return nprocs, nthreads, nqbits, times


# %%
filename = "timings_openmp.txt"
omp_nprocs, omp_nthreads, omp_nqbits, omp_times = read_dat(filename)

# filename = "timings_mpi.txt"
filename = "timings_mpi_openmp.txt"
nprocs, nthreads, nqbits, times = read_dat(filename)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4), dpi=500)
left, bottom, width, height = 0.15, 0.15, 0.8, 0.75
ax = fig.add_axes([left, bottom, width, height])
uprocs = np.sort(np.unique(nprocs))

width = 1 / (len(uprocs) - 1)
for i, p in enumerate(uprocs):
    label = f"{p:02d}-procs"
    mask = nprocs == p
    ax.bar(
        nqbits[mask] + (i - (len(uprocs) - 1) / 2) / len(uprocs) * 0.9,
        times[mask],
        width,
        align="center",
        label=label,
    )

for p, style in zip([1, 32], ["--k", ":k"]):
    label = f"{p:02d}-threads"
    mask = omp_nthreads == p
    ax.plot(omp_nqbits[mask],
            omp_times[mask],style, label=label)

ax.set_xlabel("#qubits")
ax.set_yscale("log")
ax.set_ylabel("Time [s]")
ax.set_title(f"LK-OpenMP-MPI: Time vs. #qubits")
# ax.set_xticks(np.arange(min(nqbit), max(nqbit) + 1, 2))
# ax.set_xticklabels(list(range(int(min(names)), int(max(names)) + 1)))
ax.legend(loc="best")
plt.savefig(filename.replace("txt", "png"))
