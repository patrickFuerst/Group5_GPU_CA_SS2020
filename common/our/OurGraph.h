#pragma once

#include <unordered_map>
#include <boost/functional/hash.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <random>
#include <numeric>
#include <set>
#include <cassert>	

#include <boost/filesystem.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/exterior_property.hpp>

#include <thrust/host_vector.h>

namespace fs = boost::filesystem;
using namespace boost::numeric::ublas;
using namespace boost;

class OurGraph
{

public:

	typedef std::pair<int, int> Edge;
	//typedef std::unordered_map<Edge, int, boost::hash<Edge> > EdgeMap;
	typedef std::vector<std::pair<Edge, int>> EdgeMap;

	typedef adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, int> > DirectedGraph;
	typedef graph_traits<DirectedGraph>::vertex_descriptor Vertex;
	typedef exterior_vertex_property<DirectedGraph, int> DistanceProperty;
	typedef DistanceProperty::matrix_type DistanceMatrix;

	OurGraph() {};
	OurGraph(int N, int numEdges, int weightLow) : mNumVertices(N), mNumEdges(numEdges), mWeightLow(weightLow){
		mEdgeMap.reserve(numEdges);
	};
	OurGraph(const OurGraph& g) = delete;
	OurGraph( OurGraph&& g) = default;

	matrix<int> getAdjacencyMatrix();
	thrust::host_vector<int> getAdjacencyMatrixHostVector();
	DirectedGraph getBoostGraph();
	bool checkNegativeCycles();


	//static OurGraph generateGraph(int N, float density, int weightLow, int weightHigh, unsigned seed = 1234);
	static OurGraph generateGraphOptimized(int N, float density, int weightLow, int weightHigh, unsigned seed = 1234);
	static OurGraph loadGraph(fs::path files);
	

	friend std::ostream& operator << (std::ostream& out, const OurGraph& obj)
	{

		out << "H " << obj.mNumVertices << " " << obj.mNumEdges << " " << "0\n";
		for (auto e : obj.mEdgeMap) {
			int u = e.first.first;
			int v = e.first.second;
			int weight = e.second;
			out << "E " << u << " " << v << " " << weight << "\n"; 
 		}
		return out;
	}

	friend std::istream & operator >> (std::istream & in, OurGraph& obj)
	{
		std::string line;
		in.ignore(1); // ignore first character
		int tmp;
		in >> obj.mNumVertices >> obj.mNumEdges >> tmp;
		std::getline(in, line);

		while (!in.eof()) {
			in.ignore(1); // ignore first character
			std::getline(in, line);
			int i, j, weight;
			std::stringstream ss(line);
			ss >> i >> j >> weight;

			auto newEdge = OurGraph::Edge(i, j);
			//obj.mEdgeMap.emplace(newEdge, weight);
			obj.mEdgeMap.emplace_back(std::make_pair(newEdge, weight));

		}
		return in;
	}

	EdgeMap mEdgeMap;


public:

	int mNumEdges, mNumVertices, mWeightLow;

};


matrix<int> OurGraph::getAdjacencyMatrix() {

	matrix<int> m(this->mNumVertices, this->mNumVertices, std::numeric_limits<int>::max());

	for (int i = 0; i < this->mNumVertices; i++) {
		m(i, i) = 0;
	}

	for (auto e : this->mEdgeMap) {
		int i = e.first.first;
		int j = e.first.second;
		int weight = e.second;
		m(i, j) = weight;
	}

	return m;
}

thrust::host_vector<int> OurGraph::getAdjacencyMatrixHostVector() {

	thrust::host_vector<int> m(this->mNumVertices* this->mNumVertices, std::numeric_limits<int>::max());
	for (int i = 0; i < this->mNumVertices; i++) {
		m[i  * this->mNumVertices + i ] = 0; // set the diagonal to 0
	}
	for (auto e : this->mEdgeMap) {
		int i = e.first.first;
		int j = e.first.second;
		int weight = e.second;
		m[ i * this->mNumVertices + j ] = weight;
	}

	return m;
}

OurGraph::DirectedGraph OurGraph::getBoostGraph() {

	DirectedGraph g;
	for (auto e : this->mEdgeMap) {
		int i = e.first.first;
		int j = e.first.second;
		int weight = e.second;
		boost::add_edge(i, j, weight, g);
	}
	
	return g;
}

/*
	This normal implementation uses an unordered_map as a container for the edges and weight. 
	This is very memory consuming and makes it almost impossible to generate large graphs. 

	Thus in the optimized version we switched to a vector container, which is fine for this implementation. 
	Because since we iterate over every edge ones, we don't need to check if an edge already exists and thus a vector is enough.

*/

//OurGraph OurGraph::generateGraph(int N, float density, int weightLow, int weightHigh, unsigned seed) {
//
//	unsigned int possibleNumEdges = N * (N - 1);
//	unsigned int allowedNumEdges = density * possibleNumEdges;
//	unsigned int addNumEdges = 0;
//	unsigned int notAddedVerticesCount = 0;
//
//	OurGraph graph(N, allowedNumEdges, weightLow);
//
//	std::mt19937 eng(seed); // seed the generator
//	std::uniform_int_distribution<> distrVertices(0, N - 1); // define the range
//	std::uniform_int_distribution<> distrWeights(weightLow, weightHigh); // define the range
//
//	std::vector<int> tmp(N);
//	std::iota(tmp.begin(), tmp.end(), 0); // fill with increasing numbers
//
//	std::cout << "Constructing graph with " << N << " vertices and " << allowedNumEdges << " allowed edges." << std::endl;
//
//	std::set<int> vertexNums(tmp.begin(), tmp.end());
//	tmp.clear();
//	std::cout << "Progress:";
//	for (int i = 0; i < allowedNumEdges; ) {
//
//		bool insert_success = false;
//
//		int u = distrVertices(eng);
//		int v;
//		do {
//			v = distrVertices(eng);
//		} while (u == v);
//
//		int weight = distrWeights(eng);
//		// if we conntected two vertices we can remove it 
//		vertexNums.erase(u);
//		vertexNums.erase(v);
//		auto newEdge = OurGraph::Edge(u, v);
//		auto result = graph.mEdgeMap.emplace(newEdge, weight);
//		insert_success = result.second;
//
//		if (insert_success) {
//			i++;
//			addNumEdges++;
//			// print progress
//			if (i % (allowedNumEdges / 10) == 0)
//				std::cout << float(addNumEdges) / allowedNumEdges  * 100<< "%,";
//		}
//		
//	}
//
//	std::cout << std::endl;
//	//if their are vertices left which are not connected 
//	std::cout << "Need to add " << vertexNums.size() << " to get fully connected graph." << std::endl;
//	for (auto v : vertexNums) {
//		int u = distrVertices(eng);
//		int weight = distrWeights(eng);
//		auto newEdge = OurGraph::Edge(u, v);
//		auto result = graph.mEdgeMap.emplace(newEdge, weight);
//		assert(result.second == false); // this edge should not be contained in the graph already
//		addNumEdges++;
//		
//	}
//
//	std::cout << "Constructed graph with " << addNumEdges << " edges." << std::endl;
//	//std::cout << "Graph checksum is  " << fletcher64ForMatrix(graph.getAdjacencyMatrix()) << std::endl;
//
//	return graph;
//
//}


OurGraph OurGraph::generateGraphOptimized(int N, float density, int weightLow, int weightHigh, unsigned seed) {

	unsigned int possibleNumEdges = N * (N - 1);
	unsigned int NSquared = N * N;
	unsigned int allowedNumEdges = density * possibleNumEdges;
	unsigned int addNumEdges = 0;
	unsigned int notAddedVerticesCount = 0;

	OurGraph graph(N, allowedNumEdges, weightLow);

	std::mt19937 eng(seed); // seed the generator
	std::uniform_int_distribution<> distrWeights(weightLow, weightHigh); // define the range
	std::uniform_real_distribution<double> unif(0, 1);
	std::uniform_int_distribution<> distrVertices(0, N - 1); // define the range

	std::vector<int> tmp(N);
	std::iota(tmp.begin(), tmp.end(), 0); // fill with increasing numbers
	std::set<int> vertexNums(tmp.begin(), tmp.end());
	tmp.clear();

	std::cout << "Constructing graph with " << N << " vertices and " << allowedNumEdges << " allowed edges." << std::endl;
	std::cout << "Progress:";

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {

			if (i != j) {

				bool shouldAdd = unif(eng) < density;
				if (shouldAdd) {
					auto newEdge = OurGraph::Edge(i, j);
					int weight = distrWeights(eng);
					graph.mEdgeMap.emplace_back(std::make_pair(newEdge, weight));
					addNumEdges++;
					// if we conntected two vertices we can remove it 
					vertexNums.erase(i);
					vertexNums.erase(j);
					// print progress
					if(addNumEdges % (allowedNumEdges / 10 + 1) == 0 )
						std::cout << int(float(addNumEdges) / allowedNumEdges * 100) << "%," << std::flush;

				}

			}
		}
	}
	std::cout << std::endl;


	//if their are vertices left which are not connected 
	std::cout << "Need to add " << vertexNums.size() << " to get fully connected graph." << std::endl;
	for (auto v : vertexNums) {
		int u = distrVertices(eng);
		int weight = distrWeights(eng);
		auto newEdge = OurGraph::Edge(u, v);
		graph.mEdgeMap.emplace_back(std::make_pair(newEdge, weight));
		addNumEdges++;
	}

	std::cout << "Constructed graph with " << addNumEdges << " edges." << std::endl;
	//std::cout << "Graph checksum is  " << fletcher64ForMatrix(graph.getAdjacencyMatrix()) << std::endl;

	return graph;

}


OurGraph OurGraph::loadGraph(fs::path file) {
	
	std::cout << "Loading graph!" << std::endl;

	OurGraph newGraph;
	std::ifstream in(file.string());
	in >> newGraph;
	in.close();

	std::cout << "DONE loading graph!" << std::endl;

	return newGraph;
}

bool OurGraph::checkNegativeCycles()
{
	int* dist = new int[this->mNumVertices];

	for (int i = 0; i < this->mNumVertices; i++) {
		dist[i] = INT_MAX;
	}
	dist[0] = 0;

	for (int i = 1; i < this->mNumVertices; i++) {
		for (auto e : this->mEdgeMap) {
			int src = e.first.first;
			int dest = e.first.second;
			int w = e.second;
			if (dist[src] != INT_MAX && dist[src] + w < dist[dest]) {
				dist[dest] = dist[src] + w;
			}
		}
	}

	for (auto e : this->mEdgeMap) {
		int src = e.first.first;
		int dest = e.first.second;
		int w = e.second;
		if (dist[src] != INT_MAX && dist[src] + w < dist[dest])
			return true;
	}

	return false;
}
