#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>
#include "kernels.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
using namespace std;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0
#define MAX_THREADS_PER_BLOCK 512
#define VERBOSE false
#define vprint(s...) \
	if (VERBOSE)     \
	{                \
		printf(s);   \
	}

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

	vprint("%d: Creating grid from (%f,%f) to (%f,%f) for %d points\n", level,
		   x1, y1, x2, y2, count);

	// Array of points for the geospatial data
	Point *d_points;

	// array to store the category of points (size = count) and the count of
	// points in each grid (size = 4)
	int *d_categories, *d_grid_counts;

	// Declare vectors to store the final values.
	vector<int> h_grid_counts(4);

	// Allocate memory to the pointers
	cudaMalloc(&d_points, count * sizeof(Point));
	cudaMalloc(&d_categories, count * sizeof(int));
	cudaMalloc(&d_grid_counts, 4 * sizeof(int));

	// Copy the point data into device
	cudaMemcpy(d_points, points, count * sizeof(Point), cudaMemcpyHostToDevice);

	// Set the number of blocks and threads per block
	int range, num_blocks, threads_per_block = MAX_THREADS_PER_BLOCK;
	if (count <= MAX_THREADS_PER_BLOCK)
	{
		float warps = static_cast<float>(count) / 32;
		threads_per_block = ceil(warps) * 32;
		num_blocks = 1;
	}
	else
	{
		float blocks = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
		num_blocks = min(32.0, ceil(blocks));
	}

	// Calculate the work done by each thread
	float value = static_cast<float>(count) / (num_blocks * threads_per_block);
	range = max(1.0, ceil(value));

	// KERNEL Function to categorize points into 4 subgrids
	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	vprint(
		"%d: Categorize in GPU: %d blocks of %d threads each with range=%d\n",
		level, num_blocks, threads_per_block, range);
	categorize_points<<<num_blocks, threads_per_block, 4 * sizeof(int)>>>(
		d_points, d_categories, d_grid_counts, count, range, middle_x,
		middle_y);

	// Get back the data from device to host
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

	int total = 0;
	vprint("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++)
	{
		vprint("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
		total += h_grid_counts[i];
	}
	vprint("Total Count - %d\n", count);
	if (total == count)
	{
		vprint("Sum of sub grid counts matches total point count\n");
	}

	// Declare arrays for each section of the grid and allocate memory depending
	// on the number of points found
	Point *bottom_left, *bottom_right, *top_left, *top_right;
	cudaMalloc(&bottom_left, h_grid_counts[0] * sizeof(Point));
	cudaMalloc(&bottom_right, h_grid_counts[1] * sizeof(Point));
	cudaMalloc(&top_left, h_grid_counts[2] * sizeof(Point));
	cudaMalloc(&top_right, h_grid_counts[3] * sizeof(Point));

	// KERNEL Function to assign the points to its respective array
	value = static_cast<float>(count) / threads_per_block;
	range = max(1.0, ceil(value));
	vprint("%d: Organize in GPU: 1 block of %d threads each with range=%d\n",
		   level, threads_per_block, range);
	organize_points<<<1, threads_per_block, 4 * sizeof(int)>>>(
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

	printf("\n\n");

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

	// The bounds of the grid
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

	double time_taken;
	clock_t start, end;

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	vector<QuadrantBoundary> boundaries;
	unordered_map<int, Grid *> grid_map; // maintains the grid structure of the quadrant of the given quadrant id

	for (int i = 0; i < point_count; i++)
	{
		points_array[i] = points[i];
	}
	start = clock();
	Grid *root_grid = quadtree_grid(points_array, point_count, mp(0, 0), mp(max_size, max_size), 0, nullptr, 0, boundaries, grid_map);
	end = clock();

	time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken = %lf\n\n", time_taken);

	printf("Validating grid...\n");
	pair<float, float> lower_bound = make_pair(0.0, 0.0);
	pair<float, float> upper_bound = make_pair(max_size, max_size);
	bool check = validateGrid(root_grid, upper_bound, lower_bound);

	if (check == true)
		printf("Grid Verification Success!\n");
	else
		printf("Grid Verification Failure!\n");

	vector<Query> queries = {
		{'s', Point(9981.0, 9979.0)},
		{'s', Point(5000.0, 5000.0)},
		{'i', Point(9981.0, 9979.0)},
		{'s', Point(100.0, 100.0)}
		// Add more queries as needed
	};

	// Test Search
	vector<int> results = search_quadrant(queries, boundaries);
	printf("\n\n\n @@@@@@@@ \n\n\n %d \n\n", results[2]);
	for (int i = 0; i < results.size(); i++)
	{
		printf("The point to be searched (%f, %f) with a quadrant id: %d \n \n", queries[i].point.x, queries[i].point.y, results[i]);
		if (results[i] > 0)
		{
			auto it = grid_map.find(results[i]);
			if (it != grid_map.end())
			{
				Grid *current_grid = it->second;
				bool found = false;
				for (int j = 0; j < current_grid->count; j++)
				{
					if (current_grid->points[j].x == queries[i].point.x && current_grid->points[j].y == queries[i].point.y)
					{
						found = true;
						break;
					}
				}
				printf("The type of the query is: %c \n", queries[i].type);
				switch (queries[i].type)
				{
				case 's':
					if (found)
						printf("Point found in quadrant with ID: %d\n", results[i]);
					else
						printf("Point not found in the grid.\n");
					break;
				case 'i':
					printf("Inserting a point \n");
					if (found)
						printf("Point already exists in quadrant with ID: %d\n", results[i]);
					else
						insert_point(queries[i].point, root_grid, boundaries, grid_map, results[i]);
					break;
				}
			}
			else
			{
				printf("Quadrant with ID %d not found in the map.\n", results[i]);
			}
		}
	}

	return 0;
}
