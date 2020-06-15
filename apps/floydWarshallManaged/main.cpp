#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

// CUDA runtime
#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <thrust/device_vector.h>

#include "OurGraph.h"
#include "OurHelper.h"
#include "floydWarshallManaged.h"

int main(int argc, char **argv)
{
    auto path = evaluateArgs(argc, argv);
    auto graphFiles = getGraphFiles(path);

	gpuDeviceInit(-1);


	for (auto filePath : graphFiles) {

		OurGraph graph = OurGraph::loadGraph(filePath);
		auto m = graph.getAdjacencyMatrixHostVector();

		//for (int i = 0; i < graph.mNumVertices; i++) {
		//	for (int j = 0; j < graph.mNumVertices; j++) {
		//		std::cout <<  m[i * graph.mNumVertices + j] << " ";
		//	}
		//	std::cout << std::endl;

		//}

		std::cout << " ---- START CUDA implementation ----" << std::endl;

		// Record start time
		// We actually just track the alorithm implementation 
		// not data loading 
		auto start = std::chrono::high_resolution_clock::now();


		floydWarshallManaged(m);

		// Record end time
		auto finish = std::chrono::high_resolution_clock::now();

		std::cout << " ---- END CUDA implementation ----" << std::endl;

		//for (int i = 0; i < graph.mNumVertices; i++) {
		//	for (int j = 0; j < graph.mNumVertices; j++) {
		//		std::cout << m[i * graph.mNumVertices + j] << " ";
		//	}
		//	std::cout << std::endl;

		//}


		std::chrono::duration<double, std::milli> elapsed = finish - start;
		std::cout << "Took " << elapsed.count() << " ms." << std::endl;


	}

	
}