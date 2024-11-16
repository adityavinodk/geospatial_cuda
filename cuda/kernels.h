#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <utility>
#include <unordered_map>

// namespace cg = cooperative_groups;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0

using namespace std;

struct Point
{
	float x, y;

	Point() : x(), y() {}

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
};

struct Query
{
	char type; // 'i' for insert, 's' for search
	Point point;
};

struct GridArray {
	GridArray *bottom_left, *bottom_right, *top_left, *top_right;

	int count, start_pos, grid_array_flag;

	std ::pair<float, float> top_right_corner;
	std ::pair<float, float> bottom_left_corner;

	GridArray(GridArray *bl, GridArray *br, GridArray *tl, GridArray *tr,
			  pair<float, float> uB, pair<float, float> lB, int c, int sp,
			  int gfl)
		: bottom_left(bl),
		  bottom_right(br),
		  top_left(tl),
		  top_right(tr),
		  top_right_corner(uB),
		  bottom_left_corner(lB),
		  count(c),
		  start_pos(sp),
		  grid_array_flag(gfl) {}
};

__inline__ __device__ int warpReduceSum(int value,
										cg::thread_block_tile<32> warp);

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  float middle_x, float middle_y);

__global__ void organize_points(Point *d_points, int *d_categories, Point *bl,
								Point *br, Point *tl, Point *tr, int count,
								int range);

__global__ void quadrant_search(Query *queries, int num_queries, QuadrantBoundary *boundaries, int num_boundaries, int *results);

std::vector<int> search_quadrant(const std::vector<Query> &queries, const std::vector<QuadrantBoundary> &boundaries);

void insert_point(Point new_point, Grid *root_grid, vector<QuadrantBoundary> &boundaries, unordered_map<int, Grid *> &grid_map, int quadrant_id);

void delete_point(Point point_to_delete, Grid *root_grid, vector<QuadrantBoundary> &boundaries, unordered_map<int, Grid *> &grid_map, int quadrant_id);

__global__ void reorder_points(Point *d_points, Point *grid_points,
							   int *grid_counts, int count, int range,
							   float middle_x, float middle_y, int start_pos,
							   bool opt);

// implementation for host_alloc
__global__ void reorder_points_h_alloc(Point *d_points_array,
									   Point *d_grid_points, int count,
									   int range, float middle_x,
									   float middle_y, int start_pos,
									   int *d_grid_count);

bool validate_grid(Grid *root_grid, pair<float, float> &top_right_corner,
				   pair<float, float> &bottom_left_corner);

Grid *assign_points(GridArray *root_grid, Point *grid_array1,
					Point *grid_array2);
