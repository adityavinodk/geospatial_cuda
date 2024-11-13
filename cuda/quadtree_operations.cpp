// quadtree_operations.cpp
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "kernels.h"

using namespace std;

// Function returns the quadrant id of the given point
Grid *quadtree_grid(Point *points, int count, pair<float, float> bottom_left_corner,
                    pair<float, float> top_right_corner, int level,
                    Grid *parent, int id, vector<QuadrantBoundary> &boundaries,
                    unordered_map<int, Grid *> &grid_map);

void insert_point(Point new_point, Grid *root_grid, vector<QuadrantBoundary> &boundaries, unordered_map<int, Grid *> &grid_map)
{
    // First, locate the quadrant ID where the new point should go
    int quadrant_id = search_quadrant(new_point, boundaries);
    if (quadrant_id == -1)
    {
        printf("The point is outside the grid boundaries and cannot be inserted.\n");
        return;
    }

    // Access the target grid from the unordered map
    Grid *target_grid = grid_map[quadrant_id];

    // Check if the point already exists in the quadrant
    for (int i = 0; i < target_grid->count; i++)
    {
        if (target_grid->points[i].x == new_point.x && target_grid->points[i].y == new_point.y)
        {
            printf("The point already exists in the grid.\n");
            return;
        }
    }

    // Add the point to the grid
    Point *new_points = (Point *)malloc((target_grid->count + 1) * sizeof(Point));
    memcpy(new_points, target_grid->points, target_grid->count * sizeof(Point));
    new_points[target_grid->count] = new_point;
    free(target_grid->points);
    target_grid->points = new_points;
    target_grid->count++;

    // Propagate count increment to all parent nodes
    Grid *parent_grid = target_grid->parent;
    while (parent_grid)
    {
        Point *new_points_parents = (Point *)malloc((parent_grid->count + 1) * sizeof(Point));
        memcpy(new_points_parents, parent_grid->points, parent_grid->count * sizeof(Point));
        new_points_parents[parent_grid->count] = new_point;

        free(parent_grid->points);
        parent_grid->points = new_points_parents;
        parent_grid->count++;
        parent_grid = parent_grid->parent;
    }

    // Check if the count exceeds MIN_POINTS; if so, split the quadrant
    printf("The target grid point count is: %d \n", target_grid->count);
    if (target_grid->count >= MIN_POINTS)
    {
        printf("The target grid exceeds the min point limit and needs to be further subdivided \n\n");
        vector<QuadrantBoundary> new_boundaries;
        quadtree_grid(target_grid->points, target_grid->count, target_grid->bottomLeft,
                      target_grid->topRight, 0, target_grid->parent, quadrant_id, new_boundaries, grid_map);

        // Update the boundaries in the main boundaries vector
        boundaries.insert(boundaries.end(), new_boundaries.begin(), new_boundaries.end());
    }
    printf("Point inserted successfully.\n");
}
