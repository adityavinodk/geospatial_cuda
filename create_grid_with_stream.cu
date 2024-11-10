#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "kernels.h"

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

struct Quad_point_info
{
    Point *quadrant_start_ptr;
    int count_in_quadrant;
    pair<int, int> region_bl;
    pair<int, int> region_tr;
    Quad_point_info(Point *qp, int count, pair<int, int> r_bl, pair<int, int> r_tr) : quadrant_start_ptr(qp),
                                                                                      count_in_quadrant(count),
                                                                                      region_bl(r_bl),
                                                                                      region_tr(r_tr) {}
};

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
    cudaMemcpyAsync(d_points, points, count * sizeof(Point),
                    cudaMemcpyHostToDevice, stream);

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
    cudaMemcpyAsync(h_categories.data(), d_categories, count * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

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
    cudaMemcpyAsync(bl, bottom_left, h_grid_counts[0] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(br, bottom_right, h_grid_counts[1] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tl, top_left, h_grid_counts[2] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tr, top_right, h_grid_counts[3] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);

    quad_q->push(Quad_point_info(bl, h_grid_counts[0], mp(x1, y1), mp(middle_x, middle_y)));
    quad_q->push(Quad_point_info(br, h_grid_counts[1], mp(middle_x, y1), mp(x2, middle_y)));
    quad_q->push(Quad_point_info(tl, h_grid_counts[2], mp(x1, middle_y), mp(middle_x, y2)));
    quad_q->push(Quad_point_info(tr, h_grid_counts[3], mp(middle_x, middle_y), mp(x2, y2)));

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

void build_quadtree_levels(Point *points, int point_count, queue<Quad_point_info> *quad_q, pair<int, int> bl, pair<int, int> tr)
{

    // According to GPU documentations, 32 is the limit to the number of streams 
    // but performance can not be gauranteed to be better with that many streams because of the limited number of SMs
    // We limit our streams to 4 right now
    double time_taken;
    clock_t start, end;
    start = clock();
    quadtree_grid(points, point_count, bl, tr, nullptr, quad_q);
    while (!quad_q->empty())
    {
        // start 4 streams at a time, one for each bl, br, tl, tr points
        int batch = 4;
        cudaStream_t streams[batch];
        // Initialize each stream to nullptr so that we don't get segmentation faults if we exit the grid creation early
        for (int i = 0; i < batch; ++i)
        {
            streams[i] = nullptr;
        }

        for (int i = 0; i < batch; i++)
        {
            if (quad_q->empty())
                break;

            Quad_point_info quad_point_info = quad_q->front();
            quad_q->pop();

            int x1 = quad_point_info.region_bl.fi, y1 = quad_point_info.region_bl.se,
                x2 = quad_point_info.region_tr.fi, y2 = quad_point_info.region_tr.se;
            if (!(quad_point_info.count_in_quadrant < MIN_POINTS or (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)))
            {
                cudaStreamCreate(&(streams[i]));
                printf("Stream %d created \n", i);
                quadtree_grid(quad_point_info.quadrant_start_ptr, quad_point_info.count_in_quadrant, quad_point_info.region_bl, quad_point_info.region_tr, streams[i], quad_q);
            }
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < 4; i++)
        {
            if (streams[i] != nullptr)
            {
                cudaStreamDestroy(streams[i]);
            }
        }
    }

    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);
}

int main(int argc, char *argv[])
{
    int initial_bl_fi;
    int initial_bl_se;
    int initial_tr_fi;
    int initial_tr_se;

    if (argc != 5)
    {
        fprintf(stderr, "usage: grid bl=(initial_bottom_left, initial_bottom_left) and grid tr=(initial_top_right, initial_top_right) \n");
        fprintf(stderr, "inital bottom left point must be mentioned (as a two space-sepaarted ints) \n");
        fprintf(stderr, "inital top right point must be mentioned (as a two space-sepaarted ints) \n");
        exit(1);
    }

    initial_bl_fi = (unsigned int)atoi(argv[1]);
    initial_bl_se = (unsigned int)atoi(argv[2]);
    initial_tr_fi = (unsigned int)atoi(argv[3]);
    initial_tr_se = (unsigned int)atoi(argv[4]);

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
    build_quadtree_levels(&points[0], point_count, &quad_q, mp(initial_bl_fi, initial_bl_se), mp(initial_tr_fi, initial_tr_se));
    return 0;
}
