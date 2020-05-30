

// includes, system
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "OurGraph.h"
#include "OurHelper.h"

#include <boost/numeric/ublas/io.hpp>

namespace fs =  boost::filesystem;


void floysWarshall(matrix<int>& m) {

	// Implemented according to 
	// https://dl.acm.org/doi/pdf/10.1145/367766.368168

	// This implemnetation changes the adjacency matrix in place 
	// to contain all the shortest path in the end. 
	// This saves us memory and allocation time.

	assert(m.size1() == m.size2());
	int n = m.size1();
	int inf = std::numeric_limits<int>::max();

	for (int k = 0; k < n; k++) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (m(j, k) > m(j, i) + m(i, k)) {
					int d = m(j, i) + m(i, k);
					m(j, k) = m(j, i) + m(i, k);
				}
					
			}
			
		}
	}
	// Currently the graph generation algorithm only has positive edges
	// Thus we don't need to check for negative cycley. 
	// We need to figure out how to introduce negative edges without introducing negative cycles 
	// since this is mentioned in the assignment.
	/*
	for (int i = 0; i < n; i++) {
		if (m(i, i) < 0) {
			std::cout << "Negative Circle found!";
		}

	}
	*/

}


int main(int argc, char **argv)
{


	auto path = evaluateArgs(argc, argv);
	auto graphFiles = getGraphFiles(path);


	for (auto filePath : graphFiles) {

		OurGraph graph = OurGraph::loadGraph(filePath);
		auto m = graph.getAdjacencyMatrix();


		// Record start time
		// We actually just track the alorithm implementation 
		// not data loading 
		auto start = std::chrono::high_resolution_clock::now();

		floysWarshall(m);

		// Record end time
		auto finish = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < graph.mNumVertices; i++) {
			for (int j = 0; j < graph.mNumVertices; j++) {
				std::cout << m(i,j) << " ";
			}
			std::cout << std::endl;

		}

		std::chrono::duration<double, std::milli> elapsed = finish - start;
		std::cout << "Took " << elapsed.count() << " ms." << std::endl;

	}
	exit(EXIT_SUCCESS);



}
