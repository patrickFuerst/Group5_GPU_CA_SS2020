# Group5_GPU_CA_SS2020


## Project Setup 

1. Install Cuda version 10.2 
2. Install CMake version 3.17
3. Install Boost 1.70 either by downloading and installing or directly through a package manager. 
	Description for [here](https://www.boost.org/doc/libs/1_70_0/more/getting_started/windows.html#simplified-build-from-source)

To setup a project file for Visual Studio run the following command- 

`mkdir build && cd build && cmake ../`



## TODOS

1. ~~Modify current graph generation to allow negative edges but gurantee that no negative cycles are included.~~
2. ~~Implemente a method to compare if graphs are similar.  Maybe Fletcher checksum ? http://www.cs.cornell.edu/~bindel/class/cs5220-f11/code/path.pdf~~
3. ~~Implement a loop to run the algorithms multiple time and take the mean time.~~
4. ~~Write performance results, with graphfile name, and graph checksum to a file to compare later.~~ 
5. ~~Write a script, which creates all graphs we want to test to make it reproducable for all of us.~~ 
6. ~~Write a script, which runs all test with all generated graphs.~~ 
7. Work on the other points of the assignment. 
8. Transfer Result back from device for all implementations

### Build all applications

To build all applications run the `build.sh` script. This will install all executables in the `release` folder.

### Create all graphs

To create all graphs run the `createGraphs.sh` script. This will create all graphs file in the folder `data/graphs` folder. 

### Run all benchmarks

To run all benchmarks run the `runBenchmark.sh` script. This will run all implementations with all generated graphs and save the result of each implementation to the `data/benchmarks` folder.