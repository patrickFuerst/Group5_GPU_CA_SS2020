

// includes, system
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "OurGraph.h"
#include "OurHelper.h"

#include <boost/numeric/ublas/io.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>

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

	int loopCount = 1;
	auto path = evaluateArgs(argc, argv, &loopCount);
	auto graphFiles = getGraphFiles(path);
	bool useOpenMP = true;

	std::ofstream out("./data/benchmarks/serialTimings_" + std::to_string(loopCount) + "_loops.csv");
	out << "graphFile, ourChecksum, ourTime[ms], boostTime[ms], boostChecksum" << std::endl;

	for (auto filePath : graphFiles) {

		OurGraph graph = OurGraph::loadGraph(filePath);
		double execTimings = 0;
		
		{ // scope graph matrix so it gets deleted when done and doesn't cosnume memory 
			auto m = graph.getAdjacencyMatrix();
			matrix<int> res;

			std::cout << " ---- START our serial implementation ----" << std::endl;
			#pragma omp parallel for firstprivate(m) if(useOpenMP)
			for (int i = 0; i < loopCount; i++) {
				// Record start time
				// We actually just track the alorithm implementation 
				// not data loading 
				auto start = std::chrono::high_resolution_clock::now();

				floysWarshall(m);

				// Record end time
				auto finish = std::chrono::high_resolution_clock::now();

				//for (int i = 0; i < graph.mNumVertices; i++) {
				//	for (int j = 0; j < graph.mNumVertices; j++) {
				//		std::cout << m(i, j) << " ";
				//	}
				//	std::cout << std::endl;

				//}

				if(omp_get_thread_num() == 0){
					res = m;
				}

				std::chrono::duration<double, std::milli> elapsed = finish - start;
				#pragma omp critical
				{
					std::cout << "Our implementation took " << elapsed.count() << " ms." << std::endl;
				}
				#pragma omp atomic
				execTimings += elapsed.count();

			}

			std::cout << " ---- END our serial implementation ----" << std::endl;

			std::cout << "Average time taken by ours " << execTimings / loopCount << " ms." << std::endl;

			std::string path = filePath.generic_string();

			out << path.substr(path.rfind("/") + 1) << "," << fletcher64ForMatrix(res) << "," << execTimings / loopCount << ",";
		

		}

		auto g = graph.getBoostGraph();
		OurGraph::DistanceMatrix d(graph.mNumVertices), res(graph.mNumVertices);
		boost::property_map<OurGraph::DirectedGraph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
		execTimings = 0;

		std::cout << " ---- START boost serial implementation ----" << std::endl;

		#pragma omp parallel for firstprivate(g,d,weightmap) if(useOpenMP)
		for (int i = 0; i < loopCount; i++) {
			auto start = std::chrono::high_resolution_clock::now();
			bool result = boost::floyd_warshall_all_pairs_shortest_paths(g, d, weight_map(weightmap));
			auto finish = std::chrono::high_resolution_clock::now();

			//for (int i = 0; i < graph.mNumVertices; i++) {
			//	for (int j = 0; j < graph.mNumVertices; j++) {
			//		std::cout << d[i][j] << " ";
			//	}
			//	std::cout << std::endl;

			//}

			if(omp_get_thread_num() == 0){
				res = d;
			}

			std::chrono::duration<double, std::milli> elapsed = finish - start;
			#pragma omp critical
			{
				std::cout << "Boost implementation took " << elapsed.count() << " ms." << std::endl;
			}
			#pragma omp atomic
			execTimings += elapsed.count();

		}

		std::cout << " ---- END boost serial implementation ----" << std::endl;

		std::cout << "Average time taken by boost " << execTimings / loopCount << " ms." << std::endl;

		matrix<int> resultMatrix(graph.mNumVertices, graph.mNumVertices);

		for (int i = 0; i < graph.mNumVertices; i++) {
			for (int j = 0; j < graph.mNumVertices; j++) {
				resultMatrix(i, j) = res[i][j];
			}
		}

		out << execTimings / loopCount << "," << fletcher64ForMatrix(resultMatrix) << std::endl;

	}

	out.close();

	exit(EXIT_SUCCESS);



}
