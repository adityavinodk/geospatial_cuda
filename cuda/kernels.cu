#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "kernels.h"

using namespace std;
namespace cg = cooperative_groups;

__inline__ __device__ int reduce_sum(int value,
									 cg::thread_block_tile<32> warp)
{
	// Perform warp-wide reduction using shfl_down_sync
	// Refer https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
	for (int offset = warp.size() / 2; offset > 0; offset /= 2)
	{
		value += __shfl_down_sync(0xFFFFFFFF, value, offset);
	}
	return value;
}

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  float middle_x, float middle_y)
{
	// subgrid_counts declared outside kernel, Dynamic Shared Memory
	// Accessed using extern
	extern __shared__ int subgrid_counts[];

	int start = ((blockIdx.x * blockDim.x) + threadIdx.x) * range;

	// create a thread group for 32 threads (warp grouping)
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

	// Initialize the subgrid counts to 0
	if (threadIdx.x == 0)
	{
		subgrid_counts[0] = 0;
		subgrid_counts[1] = 0;
		subgrid_counts[2] = 0;
		subgrid_counts[3] = 0;
	}
	__syncthreads();

	int first = 0, second = 0, third = 0, fourth = 0;
	for (int i = start; i < start + range; i++)
	{
		if (i < count)
		{
			// bottom left; if the point lies in bottom left, increment
			if (d_points[i].x <= middle_x and d_points[i].y <= middle_y)
			{
				d_categories[i] = 0;
				first++;
			}
			// bottom right; if point lies in bottom right, increment
			else if (d_points[i].x > middle_x and d_points[i].y <= middle_y)
			{
				d_categories[i] = 1;
				second++;
			}
			// top left; if point lies in top left, increment
			else if (d_points[i].x <= middle_x and d_points[i].y > middle_y)
			{
				d_categories[i] = 2;
				third++;
			}
			// top right; if point lies in top right, increment
			else if (d_points[i].x > middle_x and d_points[i].y > middle_y)
			{
				d_categories[i] = 3;
				fourth++;
			}
		}
	}

	// sum up all the sub quadrant counts inside a warp
	first = reduce_sum(first, warp);
	second = reduce_sum(second, warp);
	third = reduce_sum(third, warp);
	fourth = reduce_sum(fourth, warp);

	// Only the first thread in each warp writes to shared memory
	if (warp.thread_rank() == 0)
	{
		atomicAdd(&subgrid_counts[0], first);
		atomicAdd(&subgrid_counts[1], second);
		atomicAdd(&subgrid_counts[2], third);
		atomicAdd(&subgrid_counts[3], fourth);
	}
	__syncthreads();

	// Add the values of subgrid_counts to grid_counts
	if (threadIdx.x == 0)
	{
		atomicAdd(&grid_counts[0], subgrid_counts[0]);
		atomicAdd(&grid_counts[1], subgrid_counts[1]);
		atomicAdd(&grid_counts[2], subgrid_counts[2]);
		atomicAdd(&grid_counts[3], subgrid_counts[3]);
	}
}

__global__ void organize_points(Point *d_points, int *d_categories, Point *bl,
								Point *br, Point *tl, Point *tr, int count,
								int range)
{
	extern __shared__ int subgrid_index[];

	// Initialize subgrid pointer to 0
	// Used to index the point arrays for each subgrid
	if (threadIdx.x == 0)
	{
		subgrid_index[0] = 0;
		subgrid_index[1] = 0;
		subgrid_index[2] = 0;
		subgrid_index[3] = 0;
	}
	__syncthreads();

	int start = threadIdx.x * range;
	for (int i = start; i < start + range; i++)
	{
		if (i < count)
		{
			// Point array will store the respective points in a contiguous
			// fashion increment subgrid index according to the category
			unsigned int category_index =
				atomicAdd(&subgrid_index[d_categories[i]], 1);
			if (d_categories[i] == 0)
			{
				bl[category_index] = d_points[i];
			}
			if (d_categories[i] == 1)
			{
				br[category_index] = d_points[i];
			}
			if (d_categories[i] == 2)
			{
				tl[category_index] = d_points[i];
			}
			if (d_categories[i] == 3)
			{
				tr[category_index] = d_points[i];
			}
		}
	}
}

// Quandrant Search to find the level of the quadrant where the point lies
// __global__ void quadrant_search(Point *target_point, QuadrantBoundary *boundaries, int num_boundaries, int *result)
// {
// 	int idx = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (idx < num_boundaries)
// 	{
// 		QuadrantBoundary boundary = boundaries[idx];
// 		if (target_point->x >= boundary.bottom_left.first && target_point->x <= boundary.top_right.first &&
// 			target_point->y >= boundary.bottom_left.second && target_point->y <= boundary.top_right.second)
// 		{
// 			atomicMax(result, boundary.id);
// 		}
// 	}
// }

__global__ void quadrant_search(Query *queries, int num_queries, QuadrantBoundary *boundaries, int num_boundaries, int *results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_queries)
	{
		Query query = queries[idx];
		int result = -1;

		for (int i = 0; i < num_boundaries; i++)
		{
			QuadrantBoundary boundary = boundaries[i];
			if (query.point.x >= boundary.bottom_left.first && query.point.x <= boundary.top_right.first &&
				query.point.y >= boundary.bottom_left.second && query.point.y <= boundary.top_right.second)
			{
				result = max(result, boundary.id);
			}
		}

		results[idx] = result;
	}
}

// Validation Function
bool validateGrid(Grid *root_grid, pair<float, float> &TopRight, pair<float, float> &BottomLeft)
{
	if (root_grid == nullptr)
		return true;

	// If we have reached the bottom of the grid, we start validation
	if (root_grid->points)
	{
		Point *point_array = root_grid->points;
		float Top_x = TopRight.first;
		float Top_y = TopRight.second;

		float Bot_x = BottomLeft.first;
		float Bot_y = BottomLeft.second;

		float Mid_x = (Top_x + Bot_x) / 2;
		float Mid_y = (Top_y + Bot_y) / 2;

		int count = root_grid->count;

		for (int i = 0; i < count; i++)
		{
			float point_x = point_array[i].x;
			float point_y = point_array[i].y;

			if (point_x < Bot_x || point_x > Top_x)
			{
				printf("Validation Error! Point (%f, %f) is plced out of bounds. Grid dimension: [(%f, %f), (%f, %f)]\n", point_x, point_y, Bot_x, Bot_y, Top_x, Top_y);
				return false;
			}
			else if (point_y < Bot_y || point_y > Top_y)
			{
				printf("Validation Error! Point (%f, %f) is plced out of bounds. Grid dimension: [(%f, %f), (%f, %f)]\n", point_x, point_y, Bot_x, Bot_y, Top_x, Top_y);
				return false;
			}
			else
			{
				continue;
			}
		}

		return true;
	}

	// Call Recursively for all 4 quadrants
	Grid *top_left_child = nullptr;
	Grid *top_right_child = nullptr;
	Grid *bottom_left_child = nullptr;
	Grid *bottom_right_child = nullptr;

	top_left_child = root_grid->top_left;
	top_right_child = root_grid->top_right;
	bottom_left_child = root_grid->bottom_left;
	bottom_right_child = root_grid->bottom_right;

	bool check_topLeft = validateGrid(top_left_child, top_left_child->topRight, top_left_child->bottomLeft);
	bool check_topRight = validateGrid(top_right_child, top_right_child->topRight, top_right_child->bottomLeft);
	bool check_bottomLeft = validateGrid(bottom_left_child, bottom_left_child->topRight, bottom_left_child->bottomLeft);
	bool check_bottomRight = validateGrid(bottom_right_child, bottom_right_child->topRight, bottom_right_child->bottomLeft);

	return check_topLeft && check_topRight && check_bottomLeft && check_bottomRight;
}
