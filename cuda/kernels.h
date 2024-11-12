#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <utility>

struct Point
{
	float x, y;

	Point(float xc, float yc) : x(xc), y(yc) {}
};



struct Grid
{
	Grid *bottom_left, *bottom_right, *top_left, *top_right;
	Point *points;
	Grid *parent;

	// Number of points in the grid
	int count;

	// Level of the grid
	int id;

	// Grid Dimension
	std ::pair<float, float> topRight;
	std ::pair<float, float> bottomLeft;

	// Initialize the corresponding Point values
	Grid(Grid *bl, Grid *br, Grid *tl, Grid *tr, Point *ps, std ::pair<float, float> uB, std ::pair<float, float> lB, int c, Grid *p, int i)
		: bottom_left(bl),
		  bottom_right(br),
		  top_left(tl),
		  top_right(tr),
		  points(ps),
		  topRight(uB),
		  bottomLeft(lB),
		  count(c),
		  parent(p),
		  id(i) {}
};

struct QuadrantBoundary
{
	int id;
	std::pair<float, float> bottom_left;
	std::pair<float, float> top_right;
	Grid *parent;
};

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  float middle_x, float middle_y);

__global__ void organize_points(Point *d_points, int *d_categories, Point *bl,
								Point *br, Point *tl, Point *tr, int count,
								int range);

__global__ void quadrant_search(Point *target_point, QuadrantBoundary *boundaries, int num_boundaries, int *result);

bool validateGrid(Grid *root_grid, std ::pair<float, float> &TopRight, std ::pair<float, float> &BottomLeft);
