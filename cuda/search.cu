#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "kernels.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>

using namespace std;

int search_quadrant(Point target_point, const vector<QuadrantBoundary> &boundaries)
{
    QuadrantBoundary *d_boundaries;
    cudaMalloc(&d_boundaries, boundaries.size() * sizeof(QuadrantBoundary));
    cudaMemcpy(d_boundaries, boundaries.data(), boundaries.size() * sizeof(QuadrantBoundary), cudaMemcpyHostToDevice);

    Point *d_target_point;
    cudaMalloc(&d_target_point, sizeof(Point));
    cudaMemcpy(d_target_point, &target_point, sizeof(Point), cudaMemcpyHostToDevice);

    int *d_result;
    cudaMalloc(&d_result, sizeof(int));

    int init_value = -1;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = 16;
    quadrant_search<<<num_blocks, block_size>>>(d_target_point, d_boundaries, boundaries.size(), d_result);

    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_boundaries);
    cudaFree(d_target_point);
    cudaFree(d_result);

    return (result == -1) ? -1 : result; // Return -1 if point not found in any quadrant
}
