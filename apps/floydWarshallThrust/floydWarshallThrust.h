#pragma once

#include <thrust/host_vector.h>


void floydWarshallThrust(thrust::host_vector<int>& h_vec, double* copyToDeviceTimings, double* execTimings, double* copyToHostTimings);