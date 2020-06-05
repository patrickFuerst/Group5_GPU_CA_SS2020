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
	typedef std::unordered_map<Edge, int, boost::hash<Edge> > EdgeMap;
	
	typedef adjacency_list<listS, vecS, directedS, no_property, property<edge_weight_t, int> > DirectedGraph;
	typedef graph_traits<DirectedGraph>::vertex_descriptor Vertex;
	typedef exterior_vertex_property<DirectedGraph, int> DistanceProperty;
	typedef DistanceProperty::matrix_type DistanceMatrix;

	OurGraph() {};
	OurGraph(int N, int numEdges, int weightLow) : mNumVertices(N), mNumEdges(numEdges), mEdgeMap(numEdges), mWeightLow(weightLow){};
	OurGraph(const OurGraph& g) = delete;
	OurGraph( OurGraph&& g) = default;

	matrix<int> getAdjacencyMatrix();
	thrust::host_vector<int> getAdjacencyMatrixHostVector();
	DirectedGraph getBoostGraph();
	bool checkNegativeCycles();
	unsigned long fletcher64();

	static OurGraph generateGraph(int N, float density, int weightLow, int weightHigh, unsigned seed = 1234);
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
			obj.mEdgeMap.emplace(newEdge, weight);
			
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

OurGraph OurGraph::generateGraph(int N, float density, int weightLow, int weightHigh, unsigned seed) {


	int possibleNumEdges = N * (N - 1);
	int allowedNumEdges = density * possibleNumEdges;
	OurGraph graph(N, allowedNumEdges, weightLow);

	std::mt19937 eng(seed); // seed the generator
	std::uniform_int_distribution<> distrVertices(0, N-1); // define the range
	std::uniform_int_distribution<> distrWeights(weightLow, weightHigh); // define the range

	std::vector<int> tmp;
	std::iota(tmp.begin(), tmp.end(), 1); // fill with increasing numbers

	std::set<int> vertexNums(tmp.begin(), tmp.end());

	for (int i = 0; i < allowedNumEdges; i++) {

		bool edgeAlreadyInGraph = false;
		do
		{
			int u = distrVertices(eng);
			int v;
			do {
				v = distrVertices(eng);
			} while (u == v);

			int weight = distrWeights(eng);
			vertexNums.erase(u);
			vertexNums.erase(v);
			auto newEdge = OurGraph::Edge(u, v);
			auto result = graph.mEdgeMap.emplace(newEdge, weight);
			edgeAlreadyInGraph = !result.second;

		} while (edgeAlreadyInGraph);

	}

	//if their are vertices left which are not connected 
	for (auto v : vertexNums) {
		int u = distrVertices(eng);
		int weight = distrWeights(eng);
		auto newEdge = OurGraph::Edge(u, v);
		auto result = graph.mEdgeMap.emplace(newEdge, weight);
		assert(result.second == false); // this edge should not be contained in the graph already
	}

	std::cout << "Constructed graph with " << allowedNumEdges << " edges." << std::endl;

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

unsigned long OurGraph::fletcher64() {
	unsigned long sum1 = 0;
	unsigned long sum2 = 0;

	thrust::host_vector<int> dataSigned = this->getAdjacencyMatrixHostVector();
	if (mWeightLow < 0) {
		int toAdd = -mWeightLow;
		thrust::for_each(dataSigned.begin(), dataSigned.end(), [toAdd](int& i) { i == std::numeric_limits<int>::max() ? i = i : i += toAdd; });
	}
	
	thrust::host_vector<unsigned int> data = static_cast<thrust::host_vector<unsigned int>> (dataSigned);
	
	for (int i = 0; i < data.size(); ++i)
	{
		sum1 = (sum1 + data[i]) % 4294967295;
		sum2 = (sum2 + sum1) % 4294967295;

		std::cout << data[i] << std::endl;
	}

	return ((unsigned long) sum2 << 32) | sum1;
}