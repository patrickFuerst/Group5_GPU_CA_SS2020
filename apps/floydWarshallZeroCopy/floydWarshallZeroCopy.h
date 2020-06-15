#pragma once

#include <thrust/host_vector.h>

void floydWarshallZeroCopy(thrust::host_vector<int>& h_vec, double* copyTimings, double* execTimings);