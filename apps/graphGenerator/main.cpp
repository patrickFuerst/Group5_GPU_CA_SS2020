

// includes, system
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <boost/lexical_cast.hpp>
#include <boost/graph/graphviz.hpp>
#include "OurGraph.h"

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << std::endl
		<< "Options:\n"
		<< "\t-h,\tShow this help message\n"
		<< "\t-p,\tGraph parameters\n" 
		<< "\t\t\tNumber of Nodes (int)\n"
		<< "\t\t\tGraph density(0.0-1.0)\n"
		<< "\t\t\tLowest weight (int)\n"
		<< "\t\t\tHighest weight (int)\n"
		<< std::endl;
}

int main(int argc, char **argv)
{

	using boost::lexical_cast;
	using boost::bad_lexical_cast;


	int numNodes = 0;
	float density = 0.0;
	int weightRangeLow = 0;
	int weightRangeHigh = 0;

	if (argc == 1 && argv[1] == "-h") {
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc != 6) { //first argument always program name, second is -p flag
		std::cout << "Please provide 4 arguments." << std::endl;
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc == 6) {
		
		try
		{
			numNodes = lexical_cast<int>(argv[2]);
			density = lexical_cast<float>(argv[3]);
			weightRangeLow = lexical_cast<int>(argv[4]);
			weightRangeHigh = lexical_cast<int>(argv[5]);
			std::cout << "Creating graph with parameters " << std::endl
				<< "Number of Nodes: " << numNodes << std::endl
				<< "Graph density " << density << std::endl
				<< "Weight range [" << weightRangeLow << "," << weightRangeHigh << "] " << std::endl;
				 

		}
		catch (bad_lexical_cast&) {
			std::cerr << "Can not parse graph parameters!" << std::endl;
			exit(EXIT_FAILURE);
		}

	}

	// create graph 
	unsigned seed = 1234;
	auto start = std::chrono::high_resolution_clock::now();
	OurGraph graph = OurGraph::generateGraphOptimized(numNodes, density, weightRangeLow, weightRangeHigh, seed);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> total = end - start;

	std::cout << "Graph creation took: " << total.count() << std::endl;

	// check for negative cycles and create new graph if it does
	/*bool hasNegativeCycles = graph.checkNegativeCycles();
	while (hasNegativeCycles) {
		seed += 1;
		OurGraph graph = OurGraph::generateGraph(numNodes, density, weightRangeLow, weightRangeHigh, seed);
		hasNegativeCycles = graph.checkNegativeCycles();
	}*/


	// safe to file 
	std::stringstream sstm;
	sstm << "graph_" << numNodes << "_" << density << "_" << weightRangeLow << "_" << weightRangeHigh << "_" << seed;
	std::string fileName = sstm.str();

	// Write out the graph for visualization
	//std::ofstream viz_file(fileName + ".dot");
	//auto boostg = graph.getBoostGraph();
	//write_graphviz(viz_file, boostg);
	//viz_file.close();


	std::ofstream out(fileName + ".txt");
	out << graph;
	out.close();


	exit(EXIT_SUCCESS);



}
