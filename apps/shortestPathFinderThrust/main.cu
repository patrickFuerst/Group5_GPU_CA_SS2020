#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <iostream>
#include <fstream>
#include <limits>

struct functor
{
    int cur_k;
    functor(int _k) : cur_k(_k) {};
    __device__ void operator () (const int i)
    {

    }

};



int main(int argc, char **argv)
{

    if (argc != 2) {
        std::cout << "Provide filepath as argument!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream fileToOpen(argv[1]);

    if (fileToOpen) {
        std::string sanityCheck;
        int numVertices, numEdges, dummy;
        bool directed;
        fileToOpen >> sanityCheck;
        if (sanityCheck.compare("H") != 0) {
            std::cout << "Wrong file format!" << std::endl;
            exit(EXIT_FAILURE);
        }
        fileToOpen >> numVertices >> numEdges >> dummy;
        if (dummy == 0) directed = true;
        else directed = false;
        std::getline(fileToOpen, sanityCheck);

        thrust::host_vector<int> adjacencyMatrix(numVertices * numVertices, std::numeric_limits<int>::max());

        while (!fileToOpen.eof())
        {
            int source, target, weight;
            fileToOpen >> sanityCheck;
            if (sanityCheck.compare("E") != 0) {
                std::cout << "Wrong file format!" << std::endl;
                exit(EXIT_FAILURE);
            }
            fileToOpen >> source >> target >> weight;
            adjacencyMatrix[(source - 1) * numVertices + target - 1] = weight;
            std::getline(fileToOpen, sanityCheck);
        }
        fileToOpen.close();

        thrust::device_vector<int> D(adjacencyMatrix.begin(), adjacencyMatrix.end());
        
    else
    {
        std::cout << "Could not open file!" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    
    // initialize all ten integers of a device_vector to 1
    thrust::device_vector<int> D(10, 1);

    // set the first seven elements of a vector to 9
    thrust::fill(D.begin(), D.begin() + 7, 9);

    // initialize a host_vector with the first five elements of D
    thrust::host_vector<int> H(D.begin(), D.begin() + 5);

    // set the elements of H to 0, 1, 2, 3, ...
    thrust::sequence(H.begin(), H.end());

    // copy all of H back to the beginning of D
    thrust::copy(H.begin(), H.end(), D.begin());

    // print D
    for (int i = 0; i < D.size(); i++)
    {
        std::cout << "D[" << i << "] = " << D[i] << std::endl;
    }

    return 0;
}