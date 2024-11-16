#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "kernels.h"

using namespace std;

std::vector<int> search_quadrant(
	const std::vector<Query> &queries,
	const std::vector<QuadrantBoundary> &boundaries) {
	Query *d_queries;
	QuadrantBoundary *d_boundaries;
	int *d_results;

	cudaMalloc(&d_queries, queries.size() * sizeof(Query));
	cudaMalloc(&d_boundaries, boundaries.size() * sizeof(QuadrantBoundary));
	cudaMalloc(&d_results, queries.size() * sizeof(int));

	cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(Query),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_boundaries, boundaries.data(),
			   boundaries.size() * sizeof(QuadrantBoundary),
			   cudaMemcpyHostToDevice);

	int block_size = 256;
	int num_blocks = 16;
	quadrant_search<<<num_blocks, block_size>>>(
		d_queries, queries.size(), d_boundaries, boundaries.size(), d_results);

	vector<int> results(queries.size());
	cudaMemcpy(results.data(), d_results, queries.size() * sizeof(int),
			   cudaMemcpyDeviceToHost);

	cudaFree(d_queries);
	cudaFree(d_boundaries);
	cudaFree(d_results);

	return results;	 // Return -1 if point not found in any quadrant
}
