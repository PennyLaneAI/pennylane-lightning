#!/usr/bin/bash

PRECISION="double"
./benchmarks/bench_all_gates --benchmark_out="res_gates_${PRECISION}.json" --benchmark_out_format=json --benchmark_filter=".*<$PRECISION>.*"
./benchmarks/bench_all_generators --benchmark_out="res_generators_${PRECISION}.json" --benchmark_out_format=json --benchmark_filter=".*<$PRECISION>.*"
