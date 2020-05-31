
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

struct FindShorter 
{
	thrust::device_ptr <int> matrix;
	int n,k; 

	FindShorter(thrust::device_ptr <int> m, int n , int k    ) : 
		matrix(m), n(n), k(k) {}

	__host__ __device__ int operator () (int index) const
	{
		int i = index / n;
		int j = index % n;
		int oldDist = matrix[i * n + j];
		int newDist = matrix[i * n + k] + matrix[k * n + j];

		return newDist < oldDist ? newDist : oldDist;
		//return matrix[i * n + j];
	}


};


void floysWarshallThrust(thrust::host_vector<int>& h_vec)
{
	int n = sqrt(h_vec.size());
	// transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
	thrust::device_vector<int> result = h_vec;

	thrust::counting_iterator < int > c0(0);
	thrust::counting_iterator < int > c1(n*n);
	for (int k = 0; k < n; k++) {

		thrust::transform(c0, c1,result.begin(), FindShorter(d_vec.data(), n, k));

		
		thrust::swap(d_vec, result);

	}


    // transfer data back to host
    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}