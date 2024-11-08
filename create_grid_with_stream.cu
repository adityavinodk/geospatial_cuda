#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
using namespace std;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5
#define MIN_DISTANCE 5

struct Point
{
    int x, y;

    Point(int xc, int yc) : x(xc), y(yc) {}
};

struct Grid
{
    Grid *bottom_left, *bottom_right, *top_left, *top_right;
    Point *points;

    // Initialize the corresponding Point values
    Grid(Grid *bl, Grid *br, Grid *tl, Grid *tr, Point *ps)
        : bottom_left(bl),
          bottom_right(br),
          top_left(tl),
          top_right(tr),
          points(ps) {}
};

struct Quad
{
    pair<int, int> bl;
    pair<int, int> tr;
    Quad(const std::pair<int, int> &bottom_left, const std::pair<int, int> &top_right)
        : bl(bottom_left), tr(top_right) {}
};

struct Quad_point_info
{
    Point *quadrant_start_ptr;
    int count_in_quadrant;
    Quad region;
    Quad_point_info(Point *qp, int count, Quad r) : quadrant_start_ptr(qp),
                                                    count_in_quadrant(count),
                                                    region(r) {}
};

__global__ void categorize_points(Point *d_points, int *d_categories,
                                  int *grid_counts, int count, int range,
                                  int middle_x, int middle_y)
{
    // subgrid_counts declared outside kernel, Dynamic Shared Memory
    // Accessed using extern
    extern __shared__ int subgrid_counts[];

    int start = ((blockIdx.x * blockDim.x) + threadIdx.x) * range;

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

    // CUDA built in function to perform atomic addition at given location
    // Location : first variable
    // Store the counts of points in their respective subgrid
    atomicAdd(&subgrid_counts[0], first);
    atomicAdd(&subgrid_counts[1], second);
    atomicAdd(&subgrid_counts[2], third);
    atomicAdd(&subgrid_counts[3], fourth);
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

void quadtree_grid(Point *points, int count,
                   pair<int, int> bottom_left_corner,
                   pair<int, int> top_right_corner, cudaStream_t stream, queue<Quad_point_info> *quad_q)
{

    int x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
        x2 = top_right_corner.fi, y2 = top_right_corner.se;
    // subdivide points into quadrants only if we have enough points to split
    if (count < MIN_POINTS or (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE))
    {
        printf("exit condition reached \n");
        return;
    }

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
    cudaMemcpy(d_points, points, count * sizeof(Point),
               cudaMemcpyHostToDevice);

    // Set the number of blocks and threads per block
    int range, num_blocks = 16, threads_per_block = 256;

    // Calculate the work done by each thread
    float value =
        static_cast<float>(count) / (num_blocks * threads_per_block);
    range = max(1.0, ceil(value));
    printf("Categorize in GPU: %d blocks of %d threads each with range=%d\n",
           num_blocks, threads_per_block, range);

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // KERNEL Function to categorize points into 4 subgrids
    int middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
    printf("middle_x = %d, middle_y = %d \n", middle_x, middle_y);
    categorize_points<<<grid, block, 4 * sizeof(int), stream>>>(
        d_points, d_categories, d_grid_counts, count, range, middle_x,
        middle_y);

    // Get back the data from device to host
    cudaMemcpy(h_categories.data(), d_categories, count * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
               cudaMemcpyDeviceToHost);

    int total = 0;
    // printf("%d: Point counts per sub grid - \n", level);
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
    printf("Organize in GPU: 1 block of %d threads each with range=%d\n",
           threads_per_block, range);
    organize_points<<<grid2, block2, 4 * sizeof(int), stream>>>(
        d_points, d_categories, bottom_left, bottom_right, top_left, top_right,
        count, count / threads_per_block);

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

    quad_q->push(Quad_point_info(bl, h_grid_counts[0], Quad(mp(x1, y1), mp(middle_x, middle_y))));
    quad_q->push(Quad_point_info(br, h_grid_counts[1], Quad(mp(middle_x, y1), mp(x2, middle_y))));
    quad_q->push(Quad_point_info(tl, h_grid_counts[2], Quad(mp(x1, middle_y), mp(middle_x, y2))));
    quad_q->push(Quad_point_info(tr, h_grid_counts[3], Quad(mp(middle_x, middle_y), mp(x2, y2))));

    // Free data
    cudaFree(d_points);
    cudaFree(d_categories);
    cudaFree(d_grid_counts);
    cudaFree(bottom_left);
    cudaFree(bottom_right);
    cudaFree(top_left);
    cudaFree(top_right);

    return;
}

void build_quadtree_levels(Point *points, int point_count, queue<Quad_point_info> *quad_q)
{

    // limit stream to 32 but even then not gauranteed that perormance will be good becaus eof the limited number of SMs
    // limit by 4
    double time_taken;
    clock_t start, end;

    pair<int, int> initial_bl = mp(0, 0);
    pair<int, int> initial_tr = mp(1000, 1000);
    Quad initial_grid = Quad(initial_bl, initial_tr);
    start = clock();
    quadtree_grid(points, point_count, initial_bl, initial_tr, nullptr, quad_q);
    while (!quad_q->empty())
    {
        // start 4 streams at a time, one for each bl, br, tl, tr points
        cudaStream_t streams[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 4; i++)
        {
            if (quad_q->empty()) break;

            Quad_point_info quad_point_info = quad_q->front();
            quad_q->pop();

            int x1 = quad_point_info.region.bl.fi, y1 = quad_point_info.region.bl.se,
                x2 = quad_point_info.region.tr.fi, y2 = quad_point_info.region.tr.se;
            if (!(quad_point_info.count_in_quadrant < MIN_POINTS or (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)))
            {
                cudaStreamCreate(&(streams[i]));
                printf("Stream %d created \n", i);
                quadtree_grid(quad_point_info.quadrant_start_ptr, quad_point_info.count_in_quadrant, quad_point_info.region.bl, quad_point_info.region.tr, streams[i], quad_q);
            }
            else{

            }
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < 4; i++)
        {
            if(streams[i] != nullptr){
                cudaStreamDestroy(streams[i]);
            }
            
        }
    }

    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
}

int main()
{
    string filename = "points.txt";
    vector<Point> points;
    int point_count = 0;

    ifstream file(filename);
    if (!file)
    {
        cerr << "Error: Could not open the file " << filename << endl;
        return 1;
    }

    string line;
    int x, y;

    while (getline(file, line))
    {
        istringstream iss(line);
        if (iss >> x >> y)
        {
            Point p = Point(x, y);
            points.emplace_back(p);
            point_count++;
        }
        else
        {
            cerr << "Warning: Skipping malformed line: " << line << endl;
        }
    }

    file.close();
    queue<Quad_point_info> quad_q;
    build_quadtree_levels(&points[0], point_count, &quad_q);

    // quadtree_grid(points, point_count, mp(0, 0), mp(1000, 1000));

    return 0;
}
