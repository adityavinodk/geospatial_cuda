#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "kernels.h"

using namespace std;

void insert_point(Point new_point, Grid *root_grid,
				  vector<QuadrantBoundary> &boundaries,
				  unordered_map<int, Grid *> &grid_map, int quadrant_id) {
	// Access the target grid from the unordered map
	Grid *target_grid = grid_map[quadrant_id];

	// Add the point to the grid
	Point *new_points =
		(Point *)malloc((target_grid->count + 1) * sizeof(Point));
	memcpy(new_points, target_grid->points, target_grid->count * sizeof(Point));
	new_points[target_grid->count] = new_point;
	free(target_grid->points);
	target_grid->points = new_points;
	target_grid->count++;

	// Propagate count increment to all parent nodes
	Grid *parent_grid = target_grid->parent;
	while (parent_grid) {
		Point *new_points_parents =
			(Point *)malloc((parent_grid->count + 1) * sizeof(Point));
		memcpy(new_points_parents, parent_grid->points,
			   parent_grid->count * sizeof(Point));
		new_points_parents[parent_grid->count] = new_point;

		free(parent_grid->points);
		parent_grid->points = new_points_parents;
		parent_grid->count++;
		parent_grid = parent_grid->parent;
	}

	printf("Point inserted successfully.\n");
}
