# Compiler and flags
CC = nvcc
CFLAGS = -I. -g

# Target executable
TARGET = create_grid

# Source files
SRCS = create_grid.cu kernels.cu quadtree_operations.cpp

# Build rule for the executable
$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(SRCS) $(CFLAGS)

# Clean up
clean:
	rm -f $(TARGET) *~
