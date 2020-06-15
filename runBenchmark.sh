mkdir -p ./data/benchmarks

./release/floydWarshallAlgorithm -f ./data/graphs -n $1

./release/floydWarshallCuda -f ./data/graphs -n $1

./release/floydWarshallThrust  -f ./data/graphs -n $1

./release/floysWarshallZeroCopy -f ./data/graphs -n $1

./release/floysWarshallManaged -f ./data/graphs -n $1