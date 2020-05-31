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
	OurGraph(int N, int numEdges) : mNumVertices(N), mNumEdges(numEdges), mEdgeMap(numEdges){};
	OurGraph(const OurGraph& g) = delete;
	OurGraph( OurGraph&& g) = default;

	matrix<int> getAdjacencyMatrix();
	thrust::host_vector<int> getAdjacencyMatrixHostVector();
	DirectedGraph getBoostGraph();

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

	int mNumEdges, mNumVertices;

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
	OurGraph graph(N, allowedNumEdges);

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

	OurGraph newGraph;
	std::ifstream in(file.string());
	in >> newGraph;
	in.close();
	return newGraph;
}