#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <map>
#include <fstream>

#define BLOCKSIZE 256

// Doesn't do negative edges!!
__global__
void cu_FloydWarshall(int k, int *distances, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    // If we're over the edge of the matrix return
    if (col >= N) return;

    int arrayIndex = N * row + col;

    // Each block loads the entire BLOCK column into shared memory
    __shared__ int current;
    // This is done by thread #0
    if(threadIdx.x == 0)
    	current = distances[N * row + k];
    // The rest of the threads should wait
    __syncthreads();
    
    // If the current distance is INF, return
    if (current == INT_MAX / 2)
    return;

    // If the follow up distance is INF, return
    int next = distances[k * N + col];
    if(next == INT_MAX / 2)
    return;

    int candidateBetterDistance = current + next;
    if (candidateBetterDistance < distances[arrayIndex])
        distances[arrayIndex] = candidateBetterDistance;
}

int * makeWeights(int N) {
    int *weightMatrix = (int*) malloc(N * N * sizeof(int));
    // Initialize the weights to INF
    for (int ii = 0; ii < N; ii++)
        for (int jj = 0; jj < N; jj++) {
            if(ii != jj)
            	weightMatrix[ii * N + jj] = INT_MAX / 2;
            else
            	weightMatrix[ii * N + jj] = 0;
        }

    // TODO: Add the distances of the actual graph
    return weightMatrix;
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

int main(void)
{
    // TODO: Load graph!
	// Number of nodes
    int N = 4;
    // N x N distance matrix
    int *originalDistances = makeWeights(N);

    originalDistances[2] = 2;
    originalDistances[4] = 4;
    originalDistances[6] = 3;
    originalDistances[11] = 2;
    originalDistances[13] = 1;

    for (int ii = 0; ii < N; ii++) {
        for (int jj = 0; jj < N; jj++)
        	std::cout << originalDistances[ii * N + jj];
    	std::cout << std::endl;
    }

	std::cout << std::endl;

    // Transfer graph to GPU
    int *cudaDistances;   
    cudaMalloc(&cudaDistances, N * N * sizeof(int));
    cudaMemcpy(cudaDistances, originalDistances, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // For each node, iterate distances
    for (int k = 0; k < N; k++) {
    	cu_FloydWarshall <<<dim3(iDivUp(N, BLOCKSIZE), N), BLOCKSIZE>>>(k, cudaDistances, N);
    }

    // Get results back
    cudaMemcpy(originalDistances, cudaDistances, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    for (int ii = 0; ii < N; ii++) {
        for (int jj = 0; jj < N; jj++)
        	std::cout << originalDistances[ii * N + jj];
    	std::cout << std::endl;
    }
}
