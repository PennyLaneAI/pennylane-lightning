import argparse
import json
import re


def version_map(py_ver: str):
    r"""Auxiliary function to map the Python version number to the format accepted by CIBuildWheel wheel-builder actions.
    Args:
        py_ver (str): Python version number in 'major.minor' format.
    Returns:
        str: Python version in cp'major.minor'-* format.
    """
    ci_ver = re.sub("\.", "", py_ver)
    return f"cp{ci_ver}-*"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-version",
        dest="min",
        type=str,
        required=True,
        help="Minimum Python version supported",
    )
    parser.add_argument(
        "--max-version",
        dest="max",
        type=str,
        required=True,
        help="Maximum Python version supported",
    )
    parser.add_argument(
        "--range",
        dest="range",
        required=False,
        action="store_true",
        help="Include all Python values between the extrema",
    )

    args = parser.parse_args()

    v_minor_min = int(args.min.split(".")[-1])
    v_minor_max = int(args.max.split(".")[-1])

    out_range = range(v_minor_min, v_minor_max + 1) if args.range else [v_minor_min, v_minor_max]
    output_list = []

    for v in out_range:
        v_str = f"3.{v}"
        output_list.append((version_map(v_str)))

    json_out = json.dumps(output_list)

    print(json_out)
