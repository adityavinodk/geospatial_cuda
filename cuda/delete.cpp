#include <bits/stdc++.h>
#include "kernels.h"

using namespace std;

void delete_point(Point point_to_delete, Grid *root_grid, vector<QuadrantBoundary> &boundaries, unordered_map<int, Grid *> &grid_map, int quadrant_id)
{
    // Access the target grid from the unordered map
    Grid *target_grid = grid_map[quadrant_id];

    // Find and remove the point from the grid
    bool point_found = false;
    Point *new_points = (Point *)malloc((target_grid->count - 1) * sizeof(Point));
    int new_count = 0;

    for (int i = 0; i < target_grid->count; i++)
    {
        if (target_grid->points[i].x == point_to_delete.x && target_grid->points[i].y == point_to_delete.y)
        {
            point_found = true;
        }
        else
        {
            new_points[new_count++] = target_grid->points[i];
        }
    }

    if (!point_found)
    {
        free(new_points);
        printf("Point not found in the specified quadrant.\n");
        return;
    }

    // Update the grid with the new points
    free(target_grid->points);
    target_grid->points = new_points;
    target_grid->count = new_count;

    // Propagate count decrement to all parent nodes
    Grid *current_grid = target_grid;
    while (current_grid)
    {
        current_grid->count--;

        // Remove the point from the parent's point array
        if (current_grid->parent)
        {
            Point *new_parent_points = (Point *)malloc((current_grid->parent->count) * sizeof(Point));
            int new_parent_count = 0;
            for (int i = 0; i < current_grid->parent->count + 1; i++)
            {
                if (current_grid->parent->points[i].x != point_to_delete.x || current_grid->parent->points[i].y != point_to_delete.y)
                {
                    new_parent_points[new_parent_count++] = current_grid->parent->points[i];
                }
            }
            free(current_grid->parent->points);
            current_grid->parent->points = new_parent_points;
        }

        current_grid = current_grid->parent;
    }

    // Check if the count is less than MIN_POINTS
    if (target_grid->count < MIN_POINTS && target_grid->bottom_left)
    {
        printf("Removing child nodes \n");
        // Remove children nodes
        delete target_grid->bottom_left;
        delete target_grid->bottom_right;
        delete target_grid->top_left;
        delete target_grid->top_right;

        target_grid->bottom_left = nullptr;
        target_grid->bottom_right = nullptr;
        target_grid->top_left = nullptr;
        target_grid->top_right = nullptr;

        // Update boundaries vector
        boundaries.erase(
            remove_if(boundaries.begin(), boundaries.end(),
                      [quadrant_id](const QuadrantBoundary &qb)
                      {
                          return qb.id / 4 == quadrant_id;
                      }),
            boundaries.end());

        // Update grid_map
        for (auto it = grid_map.begin(); it != grid_map.end();)
        {
            if (it->first / 4 == quadrant_id && it->first != quadrant_id)
            {
                it = grid_map.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    printf("Point deleted successfully.\n");
}