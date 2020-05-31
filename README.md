# Group5_GPU_CA_SS2020


## Project Setup 

1. Install Cuda version 10.2 
2. Install CMake version 3.17
3. Install Boost 1.70 either by downloading and installing or directly through a package manager. 
	Description for [here](https://www.boost.org/doc/libs/1_70_0/more/getting_started/windows.html#simplified-build-from-source)

To setup a project file for Visual Studio run the following command- 

`mkdir build && cd build && cmake ../`



## Dokumentation 


### Graph generation 

Currently the graph generation algorithm only has positive edges.
Thus we don't need to check for negative cycle. 
We need to figure out how to introduce negative edges without introducing negative cycles since this is mentioned in the assignment.

### Floys-Warshall Algorithm

#### C++ Implementation 

Implemented according to https://dl.acm.org/doi/pdf/10.1145/367766.368168.
Our implementation changes the adjacency matrix in place to contain all the shortest path in the end. 
This saves memory and allocation time.


## TODOS

1. Modify current graph generation to allow negative edges but gurantee that no negative cycles are included. 
2. Implemente a method to compare if graphs are similar.  Maybe Fletcher checksum ? http://www.cs.cornell.edu/~bindel/class/cs5220-f11/code/path.pdf
3. Implement a loop to run the algorithms multiple time and take the mean time. 
4. Write performance results, with graphfile name, and graph checksum to a file to compare later. 
5. Write a script, which creates all graphs we want to test to make it reproducable for all of us. 
6. Write a script, which runs all test with all generated graphs. 
7. Work on the other points of the assignment. 

