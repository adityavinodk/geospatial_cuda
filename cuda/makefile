# Compiler and flags
CC = nvcc
CFLAGS = -I. -g

grid: create_grid.cu kernels.cu search.cu insert.cpp
	$(CC) -o create_grid create_grid.cu search.cu insert.cpp kernels.cu $(CFLAGS)

grid_managed: create_grid_managed.cu kernels.cu
	$(CC) -o create_grid_managed create_grid_managed.cu kernels.cu $(CFLAGS)

grid_stream: create_grid_with_stream.cu kernels.cu
	$(CC) -o create_grid_with_stream create_grid_with_stream.cu kernels.cu $(CFLAGS)

clean:
	rm -f $(TARGET) *~

