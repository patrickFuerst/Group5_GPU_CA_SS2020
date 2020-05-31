#pragma once

#include <thrust/host_vector.h>


void floydWarshallCuda(thrust::host_vector<int>& h_vec);