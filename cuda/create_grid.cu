#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "kernels.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
using namespace std;

Grid *quadtree_grid(Point *points, int count, pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int level,
					Grid *parent, int id, vector<QuadrantBoundary> &boundaries,
					unordered_map<int, Grid *> &grid_map)
{
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		  x2 = top_right_corner.fi, y2 = top_right_corner.se;

	boundaries.push_back({id, bottom_left_corner, top_right_corner});

	if (count < MIN_POINTS || (abs(x1 - x2) < MIN_DISTANCE && abs(y1 - y2) < MIN_DISTANCE))
	{
		Grid *leaf_grid = new Grid(nullptr, nullptr, nullptr, nullptr, points, {x2, y2}, {x1, y1}, count, parent, id);
		grid_map[id] = leaf_grid; // Insert the grid into the map
		return leaf_grid;
	}

	printf("%d: Creating grid from (%f,%f) to (%f,%f) for %d points\n", level,
		   x1, y1, x2, y2, count);

	// Array of points for the geospatial data
	Point *d_points;

	// array to store the category of points (size = count) and the count of
	// points in each grid (size = 4)
	int *d_categories, *d_grid_counts;

	// Declare vectors to store the final values.
	vector<int> h_categories(count);
	vector<int> h_grid_counts(4);

	// Allocate memory to the pointers
	cudaMalloc(&d_points, count * sizeof(Point));
	cudaMalloc(&d_categories, count * sizeof(int));
	cudaMalloc(&d_grid_counts, 4 * sizeof(int));

	// Copy the point data into device
	cudaMemcpy(d_points, points, count * sizeof(Point), cudaMemcpyHostToDevice);

	// Set the number of blocks and threads per block
	int range, num_blocks = 16, threads_per_block = 256;

	// Calculate the work done by each thread
	float value = static_cast<float>(count) / (num_blocks * threads_per_block);
	range = max(1.0, ceil(value));

	dim3 grid(num_blocks, 1, 1);
	dim3 block(threads_per_block, 1, 1);

	// KERNEL Function to categorize points into 4 subgrids
	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	printf("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	printf(
		"%d: Categorize in GPU: %d blocks of %d threads each with range=%d\n",
		level, num_blocks, threads_per_block, range);
	categorize_points<<<grid, block, 4 * sizeof(int)>>>(
		d_points, d_categories, d_grid_counts, count, range, middle_x,
		middle_y);

	// Get back the data from device to host
	cudaMemcpy(h_categories.data(), d_categories, count * sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

	int total = 0;
	printf("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++)
	{
		printf("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
		total += h_grid_counts[i];
	}
	printf("Total Count - %d\n", count);
	if (total == count)
	{
		printf("Sum of sub grid counts matches total point count\n");
	}

	// Declare arrays for each section of the grid and allocate memory depending
	// on the number of points found
	Point *bottom_left, *bottom_right, *top_left, *top_right;
	cudaMalloc(&bottom_left, h_grid_counts[0] * sizeof(Point));
	cudaMalloc(&bottom_right, h_grid_counts[1] * sizeof(Point));
	cudaMalloc(&top_left, h_grid_counts[2] * sizeof(Point));
	cudaMalloc(&top_right, h_grid_counts[3] * sizeof(Point));

	dim3 grid2(1, 1, 1);
	dim3 block2(threads_per_block, 1, 1);

	// KERNEL Function to assign the points to its respective array
	value = static_cast<float>(count) / threads_per_block;
	range = max(1.0, ceil(value));
	printf("%d: Organize in GPU: 1 block of %d threads each with range=%d\n",
		   level, threads_per_block, range);
	organize_points<<<grid2, block2, 4 * sizeof(int)>>>(
		d_points, d_categories, bottom_left, bottom_right, top_left, top_right,
		count, range);

	// Declare the final array in which we store the sorted points according to
	// the location in the grid
	Point *bl, *br, *tl, *tr;
	bl = (Point *)malloc(h_grid_counts[0] * sizeof(Point));
	br = (Point *)malloc(h_grid_counts[1] * sizeof(Point));
	tl = (Point *)malloc(h_grid_counts[2] * sizeof(Point));
	tr = (Point *)malloc(h_grid_counts[3] * sizeof(Point));

	// Shift the data from device to host
	cudaMemcpy(bl, bottom_left, h_grid_counts[0] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(br, bottom_right, h_grid_counts[1] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(tl, top_left, h_grid_counts[2] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(tr, top_right, h_grid_counts[3] * sizeof(Point),
			   cudaMemcpyDeviceToHost);

	// Free data
	cudaFree(d_points);
	cudaFree(d_categories);
	cudaFree(d_grid_counts);
	cudaFree(bottom_left);
	cudaFree(bottom_right);
	cudaFree(top_left);
	cudaFree(top_right);

	printf("\n");

	printf("The current parent id is: %d \n\n", id);
	// Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct
	Grid *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid = quadtree_grid(bl, h_grid_counts[0], bottom_left_corner,
							mp(middle_x, middle_y), level + 1, nullptr, id * 4, boundaries, grid_map);
	br_grid = quadtree_grid(br, h_grid_counts[1], mp(middle_x, y1),
							mp(x2, middle_y), level + 1, nullptr, id * 4 + 1, boundaries, grid_map);
	tl_grid = quadtree_grid(tl, h_grid_counts[2], mp(x1, middle_y),
							mp(middle_x, y2), level + 1, nullptr, id * 4 + 2, boundaries, grid_map);
	tr_grid = quadtree_grid(tr, h_grid_counts[3], mp(middle_x, middle_y),
							top_right_corner, level + 1, nullptr, id * 4 + 3, boundaries, grid_map);

	pair<float, float> upperBound = make_pair(x2, y2);
	pair<float, float> lowerBound = make_pair(x1, y1);

	Grid *root_grid = new Grid(bl_grid, br_grid, tl_grid, tr_grid, points, upperBound, lowerBound, count, parent, id);
	grid_map[id] = root_grid;

	if (bl_grid)
		bl_grid->parent = root_grid;
	if (br_grid)
		br_grid->parent = root_grid;
	if (tl_grid)
		tl_grid->parent = root_grid;
	if (tr_grid)
		tr_grid->parent = root_grid;

	return root_grid;
}

int main(int argc, char *argv[])
{
	if (argc < 3)
	{
		std::cerr << "Usage: " << argv[0] << " file_path max_boundary"
				  << std::endl;
		return 1;
	}

	string filename = argv[1];
	float max_size = atof(argv[2]);

	ifstream file(filename);
	if (!file)
	{
		cerr << "Error: Could not open the file " << filename << endl;
		return 1;
	}

	string line;
	float x, y;
	vector<Point> points;
	int point_count = 0;
	while (getline(file, line))
	{
		istringstream iss(line);
		if (iss >> x >> y)
		{
			Point p = Point((float)x, (float)y);
			points.emplace_back(p);
			point_count++;
		}
		else
		{
			cerr << "Warning: Skipping malformed line: " << line << endl;
		}
	}

	file.close();

	Point *points_array = points.data();
	vector<QuadrantBoundary> boundaries;
	unordered_map<int, Grid *> grid_map; // maintains the grid structure of the quadrant of the given quadrant id
	Grid *root_grid = quadtree_grid(points_array, point_count, mp(0, 0), mp(max_size, max_size), 0, nullptr, 0, boundaries, grid_map);

	// Test Search
	Point target_point(9981, 9979);

	insert_point(target_point, root_grid, boundaries, grid_map);
	int quadrant_id = search_quadrant(target_point, boundaries);

	printf("The quadrant id for the target point is: %d \n", quadrant_id);

	// Use the result to search in the specific quadrant (Need help here!)
	if (quadrant_id == -1)
	{
		printf("The point doesn't exist in the grid");
	}

	else
	{
		auto it = grid_map.find(quadrant_id);
		if (it != grid_map.end())
		{
			Grid *current_grid = it->second;
			bool found = false;
			for (int i = 0; i < current_grid->count; i++)
			{
				if (current_grid->points[i].x == target_point.x && current_grid->points[i].y == target_point.y)
				{
					found = true;
					break;
				}
			}

			if (found)
			{
				printf("Point found in quadrant with ID: %d\n", quadrant_id);
			}
			else
			{
				printf("Point not found in the grid.\n");
			}
		}
		else
		{
			printf("Quadrant with ID %d not found in the map.\n", quadrant_id);
		}
	}

	return 0;
}
