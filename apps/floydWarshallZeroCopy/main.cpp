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
#include "floydWarshallZeroCopy.h"

int main(int argc, char **argv)
{
	int loopCount = 1;
    auto path = evaluateArgs(argc, argv, &loopCount);
    auto graphFiles = getGraphFiles(path);

	std::ofstream out("./data/benchmarks/zeroCopyTimings_" + std::to_string(loopCount) + "_loops.csv");
	out << "graphFile, checksum, copyTimeHostDevice, executionTime, totalTime" << std::endl;

	gpuDeviceInit(-1);


	for (auto filePath : graphFiles) {

		OurGraph graph = OurGraph::loadGraph(filePath);
		auto m = graph.getAdjacencyMatrixHostVector();
		double copyTimings = 0, execTimings = 0, totalTimings = 0;

		//for (int i = 0; i < graph.mNumVertices; i++) {
		//	for (int j = 0; j < graph.mNumVertices; j++) {
		//		std::cout <<  m[i * graph.mNumVertices + j] << " ";
		//	}
		//	std::cout << std::endl;

		//}

		std::cout << " ---- START CUDA zero copy implementation ----" << std::endl;

		for (int i = 0; i <= loopCount; i++) {
			if (i == 1) {
				copyTimings = 0;
				execTimings = 0;
				totalTimings = 0;
			}

			// Record start time
			// We actually just track the alorithm implementation 
			// not data loading 
			auto start = std::chrono::high_resolution_clock::now();


			floydWarshallZeroCopy(m, &copyTimings, &execTimings);

			// Record end time
			auto finish = std::chrono::high_resolution_clock::now();

			std::chrono::duration<double, std::milli> total = finish - start;

			totalTimings += total.count();

		}

		std::cout << " ---- END CUDA zero copy implementation ----" << std::endl;

		std::cout << "Average copy time from host to device was " << copyTimings / loopCount << " ms." << std::endl;
		std::cout << "Average execution time was " << execTimings / loopCount << " ms." << std::endl;
		std::cout << "Average total time was " << totalTimings / loopCount << " ms." << std::endl;
		std::cout << "Checksum " << fletcher64ForVector(m) << std::endl;

		std::string path = filePath.generic_string();
		out << path.substr(path.rfind("/") + 1) << "," << fletcher64ForVector(m) << "," << copyTimings / loopCount << "," << execTimings / loopCount << "," << totalTimings / loopCount << std::endl;
	}

	out.close();
	
}