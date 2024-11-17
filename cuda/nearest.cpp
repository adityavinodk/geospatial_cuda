#include <bits/stdc++.h>
#include "kernels.h"

using namespace std;

// Custom comparator for the priority queue
struct CompareDistance
{
    Point query_point;
    CompareDistance(const Point &p) : query_point(p) {}
    bool operator()(const Point &p1, const Point &p2) const
    {
        return distance(query_point, p1) < distance(query_point, p2);
    }
};

// Helper function to calculate Euclidean distance between two points
float distance(const Point &p1, const Point &p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Helper function to find the nearest quadrant to a given point
int find_nearest_quadrant(const Point &query_point, const vector<QuadrantBoundary> &boundaries)
{
    float min_distance = FLT_MAX;
    int nearest_quadrant = -1;

    for (const auto &boundary : boundaries)
    {
        float center_x = (boundary.bottom_left.first + boundary.top_right.first) / 2;
        float center_y = (boundary.bottom_left.second + boundary.top_right.second) / 2;
        Point center(center_x, center_y);

        float dist = distance(query_point, center);
        if (dist < min_distance)
        {
            min_distance = dist;
            nearest_quadrant = boundary.id;
        }
    }

    return nearest_quadrant;
}



vector<Point> find_n_nearest_points(const Point &query_point, int n, Grid *root_grid, const vector<QuadrantBoundary> &boundaries, const unordered_map<int, Grid *> &grid_map)
{
    // Use {} instead of () for initialization
    priority_queue<Point, vector<Point>, CompareDistance> pq{CompareDistance(query_point)};
    vector<Point> result;
    unordered_set<int> visited_quadrants;

    int current_quadrant = find_nearest_quadrant(query_point, boundaries);

    while (pq.size() < n && current_quadrant != -1)
    {
        if (visited_quadrants.find(current_quadrant) != visited_quadrants.end())
        {
            break;
        }
        visited_quadrants.insert(current_quadrant);

        auto it = grid_map.find(current_quadrant);
        if (it != grid_map.end())
        {
            Grid *current_grid = it->second;
            for (int i = 0; i < current_grid->count; i++)
            {
                pq.push(current_grid->points[i]);
                if (pq.size() > n)
                {
                    pq.pop();
                }
            }
        }

        // Find the next nearest quadrant
        current_quadrant = find_nearest_quadrant(query_point, boundaries);
    }

    // Extract the n nearest points from the priority queue
    while (!pq.empty() && result.size() < n)
    {
        result.push_back(pq.top());
        pq.pop();
    }

    return result;
}