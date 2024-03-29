#pragma once

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "OurGraph.h"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace fs = boost::filesystem;

static void show_usage(std::string name)
{
	std::cerr << "Usage: " << name << std::endl
		<< "Options:\n"
		<< "\t-h,\tShow this help message\n"
		<< "\t-f,\tFile containing graph data\n"
		<< "\t\t\tIf it's a directory the algorithm is run on all files.\n"
		<< "\t-n,\t Defines how often the algorithm should be executed per graph.\n"
		<< std::endl;
}


using boost::lexical_cast;
using boost::bad_lexical_cast;

fs::path evaluateArgs(int argc, char** argv, int* counter) {
	fs::path dataPath = "";

	if (argc == 1 && argv[1] == "-h") {
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc != 5) { //first argument always program name, second is -f flag, -n is fourth,
		std::cout << "Please provide the path to the file(s) and the number of loops to be executed." << std::endl;
		show_usage(argv[0]);
		exit(EXIT_SUCCESS);

	}
	else if (argc == 5) {

		dataPath = argv[2];

		*counter = std::stoi(argv[4]);

		if (fs::exists(dataPath))
		{
			if (fs::is_directory(dataPath)) {
				return dataPath;
			}
			else if (fs::is_regular_file(dataPath) && dataPath.extension() == ".txt") {
				return dataPath;
			}
			else {
				std::cout << "Please provide a valid path to the file(s)." << std::endl;
				exit(EXIT_SUCCESS);
			}

		}else {
			std::cout << "Please provide a correct path to the file(s)." << std::endl;
			exit(EXIT_SUCCESS);
		}


	}

}


std::vector<fs::path> getGraphFiles(fs::path dataPath) {
	
	std::vector<fs::path> paths;

	if (fs::is_directory(dataPath)) {
		fs::directory_iterator endIter;
		for (fs::directory_iterator iter(dataPath); iter != endIter; ++iter) {
			if (fs::is_regular_file(iter->path()) && iter->path().extension() == ".txt") {
				paths.push_back(iter->path());
			}
		}
	}else if (fs::is_regular_file(dataPath) && dataPath.extension() == ".txt") {
		paths.push_back(dataPath);
	}

	if (paths.size() == 0) {
		std::cout << "No valid graph files found!" << std::endl;
	}

	return paths;

}

unsigned long long fletcher64(std::vector<int> v) {

	// comment in if checksum of sorted vector is desired
	//std::sort(v.begin(), v.end());

	unsigned long long sum1 = 0, sum2 = 0;

	for (int i = 0; i < v.size(); i++) {
		sum1 = (sum1 + v[i]) % 4294967295;
		sum2 = (sum2 + sum1) % 4294967295;
	}


	return (sum2 << 32) | sum1;
}



unsigned long long fletcher64ForMatrix(matrix<int> m) {
	assert(m.size1() == m.size2());
	int n = m.size1();
	
	std::vector<int> v;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			v.push_back(m(i, j));
		}
	}

	return fletcher64(v);

}

unsigned long long fletcher64ForVector(thrust::host_vector<int> m) {
	std::vector<int> v;
	
	for (int i = 0; i < m.size(); i++) {
		v.push_back(m[i]);
	}

	return fletcher64(v);
}

