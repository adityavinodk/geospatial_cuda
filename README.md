# Geospatial Data + CUDA
This repository consists of code created for constructing quadtrees and performing inference on them using CUDA.

## Generating Random Points
In order to generate random points, do the following - 
Run `python generate_points.py point_count boundary_size` where `point_count` is the maximum number of points and `boundary_size` is the boundary size for the grid. 

This will assume that the points will be randomly generated between `(0, 0)` and `(boundary_size, boundary_size)`. The points will be output into a file `points.txt`

## Run the Quadtree Construction Code
In order to construct a Quadtree with the most optimal GPU code - 
1. Run `cd cuda/`
2. Run `make grid_opt`. This step assumes that you have a GPU connected to your system and have `nvcc` installed 
3. Run `./grid_opt points_file_path boundary_size`. For example run `./grid_opt ../points.txt 1000000`

## File Heirarchy
The `cuda/` directory consists of a number of different implementations of quadtree construction code and inferencing. They can be compiled using the makefile. Here are some relevant files - 
1. `create_grid_opt_v2.cu` - Consists of the fastest quadtree construction algorithm with a single kernel using 2 array strategy for recursive calls. Compile using `make grid_opt`
2. `create_grid.cu` - Simpler implementation of quadtree construction using 2 kernels for categorizing and organizing. Compile using `make grid`
3. `inference.cu` - Implementation of running search, insert and delete queries on the Grid structure with kernel for searching boundaries. Compile using `make grid_inference`. 
    - To generate queries, run `python generate_queries num_queries boundary_size`, which will generate a `queries.txt` file 
    - To run the inference, run the following - `./grid_inference points_file boundary_size queries_file`
4. The code for streams and OpenMP have been implemented as `streams_host_alloc.cu` (compile with `make stream_alloc`) and `streams_omp.cu` (compile with `make stream_omp`) files within the [streams_omp](https://github.com/adityavinodk/geospatial_cuda/tree/streams_omp) branch. This work is still under development.

The `sequential/` directory consists of the code for sequential quadtree construction on CPU (without GPU). This was created mainly for benchmarking purposes and testing performance of the GPU code.

## Future Work
1. Build scalable OpenMP based Quadtree construction 
2. Quadtree construction using Dynamic Parallelism for recursive grid construction
3. Optimizing Queries by grouping based on operation and boundaries for faster performance