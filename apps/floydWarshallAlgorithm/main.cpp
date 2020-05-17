

// includes, system
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "OurGraph.h"
#include <boost/numeric/ublas/io.hpp>


namespace fs =  boost::filesystem;

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << std::endl
		<< "Options:\n"
		<< "\t-h,\tShow this help message\n"
		<< "\t-f,\tFile containing graph data\n" 
		<< "\t\t\tIf it's a directory the algorithm is run on all files.\n"
		<< std::endl;
}


void floysWarshall(matrix<int>& m) {

	// Implemented according to 
	// https://dl.acm.org/doi/pdf/10.1145/367766.368168

	// This implemnetation changes the adjacency matrix in place 
	// to contain all the shortest path in the end. 
	// This saves us memory and allocation time.

	assert(m.size1() == m.size2());
	int n = m.size1();
	int inf = std::numeric_limits<int>::max();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (m(j, i) < inf) {
				for (int k = 0; k < n; k++) {
					if (m(i, k) < inf) {
						if (m(j, k) > m(j, i) + m(i, k)) {
							int d = m(j, i) + m(i, k);
							m(j, k) = m(j, i) + m(i, k);
						}
					}
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

	using boost::lexical_cast;
	using boost::bad_lexical_cast;

	bool evaluateMultipleFiles = false;
	fs::path dataPath = "";

	if (argc == 1 && argv[1] == "-h") {
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc != 3) { //first argument always program name, second is -f flag
		std::cout << "Please provide the path to the file(s)." << std::endl;
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc == 3) {
		
	
			
		dataPath  = argv[2];

		if(fs::exists(dataPath) )
		{
			if(fs::is_directory(dataPath) ) {
				evaluateMultipleFiles = true;
			}
			else if (!fs::is_regular_file(dataPath)) {
				std::cout << "Please provide a valid path to the file(s)." << std::endl;
				exit(EXIT_SUCCESS);
			}
				
		}
	
	}

	if (evaluateMultipleFiles) {

		fs::directory_iterator endIter; 
		for (fs::directory_iterator iter(dataPath); iter != endIter; ++iter) {

			if (fs::is_regular_file(iter->path()) && iter->path().extension() == ".txt") {

				std::cout << "Processing file " << iter->path().filename() << std::endl;

				OurGraph graph = OurGraph::loadGraph(iter->path());
				auto m = graph.getAdjacencyMatrix();
				floysWarshall(m);

			}


		}

	}


	exit(EXIT_SUCCESS);



}
