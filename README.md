# Group5_GPU_CA_SS2020


## Project Setup

1. Install Cuda version 10.2
2. Install CMake version 3.17
3. Install Boost 1.70 either by downloading and installing or directly through a package manager.
	Description for [here](https://www.boost.org/doc/libs/1_70_0/more/getting_started/windows.html#simplified-build-from-source)

To setup a project files run the following command. This will create platform specific project files.

`mkdir build && cd build && cmake ../`



## TODOS

1. ~~Modify current graph generation to allow negative edges but gurantee that no negative cycles are included.~~
2. ~~Implemente a method to compare if graphs are similar.  Maybe Fletcher checksum ? http://www.cs.cornell.edu/~bindel/class/cs5220-f11/code/path.pdf~~
3. ~~Implement a loop to run the algorithms multiple time and take the mean time.~~
4. ~~Write performance results, with graphfile name, and graph checksum to a file to compare later.~~
5. ~~Write a script, which creates all graphs we want to test to make it reproducable for all of us.~~
6. ~~Write a script, which runs all test with all generated graphs.~~
7. ~~Work on the other points of the assignment.~~
8. ~~Transfer Result back from device for all implementations~~

### Build all applications

To build all applications run the `build.sh` script. This will install all executables in the `release` folder.

### Create all graphs

To create all graphs run the `createGraphs.sh` script. This will create all graphs file in the folder `data/graphs` folder.
Currently, the largest graph is specified with 20.000 nodes. Running this script will take around 45 minutes until all graphs
are created.

In order to reduce the number of nodes the variable `num_nodes` containing all node numbers within the `createGraphs.sh` script can be edited.
Or the program `graphGenerator` within the release folder can be called with the right parameters.

### Run all benchmarks

To run all benchmarks run the `runBenchmark.sh` script with the parameters `-n k`, where *k* defines how often each implementation is run to calculate
the mean execution time. This will run all implementations with all generated graphs and save the result of each implementation to the `data/benchmarks` folder.

**!! THIS SCRIPT RUNS VERY LONG !!**

If this is called with graphs containing 20.000 nodes this will take days, because of the serial implementation being so slow.

In order to call one implementation with one graph file call the appropriate application from the `release` folder and specifiy the path to the graph file
and how often it should run it like so:

`./floysWahrshallApp -f ../data/graphs/graphfile.txt -n 1`
