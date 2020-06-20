
#include "floydWarshallCuda.h"
#include <thrust/device_vector.h>
#include <chrono>


#define BLOCKSIZE 256

// Doesn't do negative edges!!
__global__
void iterKernelNoShared(int k, int *distances, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    // If we're over the edge of the matrix return
    if ((col >= N) || (distances[N * row + k] == INT_MAX) || (distances[k * N + col] == INT_MAX))
        return;

    // Otherwise, calculate the distance
    int candidateBetterDistance = distances[N * row + k] + distances[k * N + col];
    if (candidateBetterDistance < distances[N * row + col])
        distances[N * row + col] = candidateBetterDistance;
}

// Doesn't do negative edges!!
__global__
void iterKernelShared(int k, int *distances, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    // If we're over the edge of the matrix return
    if (col >= N) 
        return;

    int arrayIndex = N * row + col;

    // Each block loads the entire BLOCK column into shared memory
    __shared__ int current;
    // This is done by thread #0
    if(threadIdx.x == 0)
    	current = distances[N * row + k];
    // The rest of the threads should wait
    __syncthreads();
    
    // If the current distance is INF, return
    if (current == INT_MAX )
        return;

    // If the follow up distance is INF, return
    int next = distances[k * N + col];
    if(next == INT_MAX )
        return;

    // Otherwise, calculate the distance
    int candidateBetterDistance = current + next;
    if (candidateBetterDistance < distances[arrayIndex])
        distances[arrayIndex] = candidateBetterDistance;
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void floydWarshallCuda(thrust::host_vector<int>& h_vec, double* copyToDeviceTimings, double* execTimings, double* copyToHostTimings)
{
    // Track subtask time
	auto timeInit = std::chrono::high_resolution_clock::now();
    
    // Transfer graph to GPU
    int N = sqrt(h_vec.size());

    thrust::device_vector<int> d_vec = h_vec;
    thrust::device_ptr<int> d_ptr = d_vec.data();
    
    // Device memory allocated
	auto timeHtD = std::chrono::high_resolution_clock::now();

    // For each node, iterate distances
    for (int k = 0; k < N; k++)
        iterKernelShared<<< dim3(iDivUp(N, BLOCKSIZE), N), BLOCKSIZE >>> ( k, thrust::raw_pointer_cast(d_ptr), N );

    cudaDeviceSynchronize();

    // Calculations complete
	auto timeExec = std::chrono::high_resolution_clock::now();

    // Get results back
    h_vec = d_vec;

    // Results moved to host
    auto timeDtH = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> hostToDevice = timeHtD - timeInit;
    *copyToDeviceTimings += hostToDevice.count();

    std::chrono::duration<double, std::milli> exec = timeExec - timeHtD;
    *execTimings += exec.count();

    std::chrono::duration<double, std::milli> deviceToHost = timeDtH - timeExec;
    *copyToHostTimings += deviceToHost.count();
}