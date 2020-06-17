#!/bin/sh

NUM_RUNS=$1

mkdir -p ./data/benchmarks

./release/floydWarshallAlgorithm -f ./data/graphs -n $NUM_RUNS

./release/floydWarshallCuda -f ./data/graphs -n $NUM_RUNS

./release/floydWarshallThrust  -f ./data/graphs -n $NUM_RUNS

./release/floydWarshallZeroCopy -f ./data/graphs -n $NUM_RUNS

./release/floydWarshallManaged -f ./data/graphs -n $NUM_RUNS