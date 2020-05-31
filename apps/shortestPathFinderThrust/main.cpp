#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>
#include <fstream>
#include <limits>

#include "floydWarshallThrust.h"
#include "OurHelper.h"

int main(int argc, char **argv)
{
    std::cout << "Thrust v " << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl;

    auto path = evaluateArgs(argc, argv);
    auto graphFiles = getGraphFiles(path);


	for (auto filePath : graphFiles) {

		OurGraph graph = OurGraph::loadGraph(filePath);
		auto m = graph.getAdjacencyMatrixHostVector();

		//for (int i = 0; i < graph.mNumVertices; i++) {
		//	for (int j = 0; j < graph.mNumVertices; j++) {
		//		std::cout <<  m[i * graph.mNumVertices + j] << " ";
		//	}
		//	std::cout << std::endl;

		//}

		std::cout << " ---- START Thrust implementation ----" << std::endl;

		// Record start time
		// We actually just track the alorithm implementation 
		// not data loading 
		auto start = std::chrono::high_resolution_clock::now();


		floysWarshallThrust(m);
		// print sorted array
		
		// Record end time
		auto finish = std::chrono::high_resolution_clock::now();

		std::cout << " ---- END Thrust implementation ----" << std::endl;

		//for (int i = 0; i < graph.mNumVertices; i++) {
		//	for (int j = 0; j < graph.mNumVertices; j++) {
		//		std::cout << m[i * graph.mNumVertices + j] << " ";
		//	}
		//	std::cout << std::endl;

		//}
		//thrust::copy(m.begin(), m.end(), std::ostream_iterator<int>(std::cout, "\n"));



		std::chrono::duration<double, std::milli> elapsed = finish - start;
		std::cout << "Took " << elapsed.count() << " ms." << std::endl;


	}

	
}