# Compiler and flags
CC = nvcc
CFLAGS = -I. -g

# Target executable
TARGET = create_grid

# Source files
SRCS = create_grid.cu kernels.cu search.cu insert.cpp

# Build rule for the executable
$(TARGET): $(SRCS)
	$(CC) -o $(TARGET) $(SRCS) $(CFLAGS)

# Clean up
clean:
	rm -f $(TARGET) *~

