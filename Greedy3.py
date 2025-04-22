# CS 3100
# Greedy-3 Approximation Algorithm
#

import time
import random
import csv

def load_latin_square_from_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        return [[int(cell) for cell in row] for row in reader]

def is_valid(grid, row, col, symbol):
    n = len(grid)
    for i in range(n):
        if grid[row][i] == symbol or grid[i][col] == symbol:
            return False
    return True

def greedy_latin_completion(grid):
    n = len(grid)
    start = time.perf_counter()
    filled = 0

    # List of empty cells
    empty = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 0]
    random.shuffle(empty)  # for conflicts

    for row, col in empty:
        for symbol in range(1, n + 1):
            if is_valid(grid, row, col, symbol):
                grid[row][col] = symbol
                filled += 1
                break

    end = time.perf_counter()
    return grid, filled, end - start

def generate_partial_latin_square(n, fill_percent=0.5):
    grid = [[0 for _ in range(n)] for _ in range(n)]
    symbols = list(range(1, n + 1))

    for i in range(n):
        row_symbols = symbols.copy()
        random.shuffle(row_symbols)
        for j in range(n):
            if random.random() < fill_percent:
                if row_symbols[j] not in grid[i] and all(row_symbols[j] != grid[k][j] for k in range(n)):
                    grid[i][j] = row_symbols[j]
    return grid

def run_trials(algorithm_fn, grid, runs=5):
    filled_list = []
    time_list = []

    for _ in range(runs):
        trial_grid = [row[:] for row in grid]  # Deep copy
        _, filled, t = algorithm_fn(trial_grid)
        filled_list.append(filled)
        time_list.append(t)

    avg_filled = sum(filled_list) / runs
    avg_time = sum(time_list) / runs
    return avg_filled, avg_time


if __name__ == "__main__":
    dataset_files = {
        "5x5": "dataset1_5x5.csv",
        "10x10": "dataset2_10x10.csv",
        "15x15": "dataset3_15x15.csv"
    }

    for label, filename in dataset_files.items():
        print(f"\nRunning for dataset {label}...")
        grid = load_latin_square_from_csv(filename)
        avg_filled, avg_time = run_trials(greedy_latin_completion, grid)
        print(f"Avg. filled: {avg_filled:.2f} / {len(grid) ** 2}")
        print(f"Avg. time: {avg_time:.4f} sec")
