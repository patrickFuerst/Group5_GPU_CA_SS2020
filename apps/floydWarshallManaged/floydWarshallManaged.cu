#include "floydWarshallManaged.h"
#include <thrust/device_vector.h>
#include <chrono>

#include <cuda.h>

#define BLOCKSIZE 256

// Doesn't do negative edges!!
__global__
void iterKernel(int k, int *distances, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;

    // If we're over the edge of the matrix return
    if ((col >= N) || (distances[N * row + k] == INT_MAX) || (distances[k * N + col] == INT_MAX)) {
        return;
    }

    // Otherwise, calculate the distance
    int candidateBetterDistance = distances[N * row + k] + distances[k * N + col];
    if (candidateBetterDistance < distances[N * row + col])
        distances[N * row + col] = candidateBetterDistance;
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void floydWarshallManaged(thrust::host_vector<int>& h_vec, double* copyTimings, double* execTimings)
{
    int N = sqrt(h_vec.size());
    int size = N * N * sizeof(int);

    // Track subtask time
    auto timeInit = std::chrono::high_resolution_clock::now();
    
    int* cudaData;
    cudaMallocManaged(&cudaData, size);

    for (int i = 0; i < h_vec.size(); i++)
        cudaData[i] = h_vec[i];

    // Device memory allocated
    auto timeHtD = std::chrono::high_resolution_clock::now();
    
    for (int k = 0; k < N; k++) 
        iterKernel<<< dim3(iDivUp(N, BLOCKSIZE), N), BLOCKSIZE >>> (k, cudaData, N);

    cudaDeviceSynchronize();

    // Calculations complete
	auto timeExec = std::chrono::high_resolution_clock::now();

    // Transfer data back to host
    for (int i = 0; i < h_vec.size(); i++)
        h_vec[i] = cudaData[i];

    auto timeDtH = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> hostToDevice = timeHtD - timeInit;
    std::cout << "Copying data from host to device took " << hostToDevice.count() << " ms." << std::endl;
    *copyTimings += hostToDevice.count();

    std::chrono::duration<double, std::milli> exec = timeExec - timeHtD;
    std::cout << "Executing calculations took " << exec.count() << " ms." << std::endl;
    *execTimings += exec.count();

    std::chrono::duration<double, std::milli> deviceToHost = timeDtH - timeExec;
    std::cout << "Copying data from device to host took " << deviceToHost.count() << " ms." << std::endl;
    *copyTimings += deviceToHost.count();
}