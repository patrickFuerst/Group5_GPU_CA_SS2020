
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <chrono>

struct FindShorter 
{
	thrust::device_ptr <int> matrix;
	int n,k; 
	int inf = std::numeric_limits<int>::max();

	FindShorter(thrust::device_ptr <int> m, int n , int k    ) : 
		matrix(m), n(n), k(k) {}

	__host__ __device__ int operator () (int index) const
	{
		int i = index / n;
		int j = index % n;
		int oldDist = matrix[i * n + j];

		int ik = matrix[i * n + k];
		int jk = matrix[k * n + j];

		if (ik == inf || jk == inf)
			return oldDist;
		
		int newDist = ik + jk;
		return newDist < oldDist ? newDist : oldDist;
	}


};


void floydWarshallThrust(thrust::host_vector<int>& h_vec, double* copyToDeviceTimings, double* execTimings, double* copyToHostTimings)
{
	int n = sqrt(h_vec.size());

	// Track subtask time
	auto timeInit = std::chrono::high_resolution_clock::now();

	// transfer data to the device
    thrust::device_vector<int> d_vec = h_vec;
	thrust::device_vector<int> result = h_vec;

	thrust::counting_iterator < int > c0(0);
	thrust::counting_iterator < int > c1(n*n);
	
	// Device memory allocated
	auto timeHtD = std::chrono::high_resolution_clock::now();

	for (int k = 0; k < n; k++) {
		thrust::transform(c0, c1,result.begin(), FindShorter(d_vec.data(), n, k));
		thrust::swap(d_vec, result);
	}

	// Calculations complete
	auto timeExec = std::chrono::high_resolution_clock::now();

	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

	auto timeDtH = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> hostToDevice = timeHtD - timeInit;
	*copyToDeviceTimings += hostToDevice.count();

	std::chrono::duration<double, std::milli> exec = timeExec - timeHtD;
	*execTimings += exec.count();

	std::chrono::duration<double, std::milli> deviceToHost = timeDtH - timeExec;
	*copyToHostTimings += deviceToHost.count();
}