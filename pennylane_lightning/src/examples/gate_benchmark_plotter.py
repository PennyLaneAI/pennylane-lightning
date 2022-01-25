import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Needed to read .csv file

if __name__ == "__main__":
    assert len(sys.argv) == 3, (
        "Usage: $PYTHON3_PATH " + sys.argv[0] + " $PATH_TO_CSV $PATH_TO_COMPILER_INFO"
    )

    data_df = pd.read_csv(sys.argv[1])
    num_qubits_idx = data_df.columns.get_loc("Num Qubits")
    time_idx = data_df.columns.get_loc(" Time (milliseconds)")

    compiler_info = open(sys.argv[2], "r").readlines()
    optimization = "-O3"

    data = data_df.to_numpy()
    avg_time_arr = [
        np.average(data[data[:, num_qubits_idx] == num_qubits][:, time_idx])
        for num_qubits in data[:, num_qubits_idx]
    ]

    # Plot absolute values in lin-lin plot
    plt.title("Averaged Absolute Time vs Number of Qubits\n")
    plt.xlabel("Number of Qubits in $[1]$")
    plt.ylabel("Time in $[ms]$")
    plt.grid(linestyle=":")
    plt.plot(data[:, num_qubits_idx], avg_time_arr, "rX")
    plt.figtext(
        0.05,
        0.0,
        ("Compiler:\t" + compiler_info[0] + "Optimization:\t" + optimization).expandtabs(),
        fontsize=7,
        va="bottom",
        ha="left",
    )
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("avg_time.png", dpi=200)
    plt.close()

    # Plot relative values in log-log plot
    plt.title("Scaling Behaviour: Relative Time vs Number of Qubits")
    plt.xlabel("Number of Qubits in $[1]$")
    plt.ylabel("Relative Time (compared to 1 qubit) in $[1]$")
    plt.grid(linestyle=":")
    plt.loglog(data[:, num_qubits_idx], avg_time_arr / avg_time_arr[0], "rX")
    plt.figtext(
        0.05,
        0.0,
        ("Compiler:\t" + compiler_info[0] + "Optimization:\t" + optimization).expandtabs(),
        fontsize=7,
        va="bottom",
        ha="left",
    )
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("scaling.png", dpi=200)
    plt.close()
