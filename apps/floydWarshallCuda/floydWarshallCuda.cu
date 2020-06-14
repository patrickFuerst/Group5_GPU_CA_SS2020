
#include "floydWarshallCuda.h"
#include <thrust/device_vector.h>


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



int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void floydWarshallCuda(thrust::host_vector<int>& h_vec)
{
    // Track subtask time
	auto timeInit = std::chrono::high_resolution_clock::now();
    
    // Transfer graph to GPU
    //int matrixSize = graph.mNumVertices * graph.mNumVertices;
    //int* cudaDistances;
    //cudaMalloc(&cudaDistances, matrixSize * sizeof(int));
    //cudaMemcpy(cudaDistances, m.data, matrixSize * sizeof(int), cudaMemcpyHostToDevice);
    int N = sqrt(h_vec.size());

    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_ptr< int > d_ptr = d_vec.data();
    
    // Device memory allocated
	auto timeHtD = std::chrono::high_resolution_clock::now();

    // For each node, iterate distances
    for (int k = 0; k < N; k++) {
        cu_FloydWarshall<<< dim3(iDivUp(N, BLOCKSIZE), N), BLOCKSIZE >>> ( k, thrust::raw_pointer_cast(d_ptr), N );
    }

    // Calculations complete
	auto timeExec = std::chrono::high_resolution_clock::now();

    // Get results back
    h_vec = d_vec;

    // Results moved to host
    auto timeDtH = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> hostToDevice = timeHtD - timeInit;
    std::cout << "Copying data from host to device took " << hostToDevice.count() << " ms." << std::endl;

    std::chrono::duration<double, std::milli> exec = timeExec - timeHtD;
    std::cout << "Executing calculations took " << exec.count() << " ms." << std::endl;

    std::chrono::duration<double, std::milli> deviceToHost = timeDtH - timeExec;
    std::cout << "Copying results from device to host took " << deviceToHost.count() << " ms." << std::endl;
}