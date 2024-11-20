import random

def generate_random_queries(filename, num_queries, grid_size=(1e6, 1e6)):
    queries = ['s', 'i', 'd']  # search, insert, delete
    operations = []
    for _ in range(int(num_queries)):
        query_type = random.choice(queries)
        x = random.randint(0, grid_size[0]-1)
        y = random.randint(0, grid_size[1]-1)
        operations.append(f'{query_type} {x:.6f} {y:.6f}')

    with open(filename, 'w') as file:
        for operation in operations:
            file.write(operation + '\n')

def main():
    max_queries = 100  # Specify how many queries (adjust as needed)
    filename = 'queries.txt'
    grid_size = (1e6, 1e6)  # Maximum grid size

    generate_random_queries(filename, max_queries, grid_size)
    print(f'Generated {max_queries} random queries in {filename}')

if __name__ == '__main__':
    main()