import json
import argparse
import re

def version_map(py_ver):
    ci_ver = re.sub('\.', '', py_ver)
    return f"cp{ci_ver}-*"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-version", dest = "min", type=str, required=True, help="Minimum Python version supported")
    parser.add_argument("--max-version", dest = "max", type=str, required=True, help="Maximum Python version supported")
    parser.add_argument("--range", dest = "range", required=False, action='store_true', help="Include all Python values between the extrema")

    args = parser.parse_args()
    output = { "cibw_build" : [version_map(args.min), version_map(args.max)] }

    if args.range:
        minor_min = int(args.min.split(".")[-1])
        minor_max = int(args.max.split(".")[-1])
        for v in range(minor_min+1, minor_max):
            v_str = f"3.{v}"
            output["cibw_build"].append(version_map(v_str) )

    print(json.dumps(output))
