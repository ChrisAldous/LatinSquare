import copy
import csv
import random
import sys
import numpy as np
import time
sys.setrecursionlimit(20000) 
def setup_square(square):
    latin_square = copy.deepcopy(square)
    all_symbols = {str(i) for i in range(1, len(square)+1)}
    n = len(latin_square)
    # getting all symbols already within each row and removing them from the set of all symbols to fill in '_'
    for row in range(n):
        temp_set = set(all_symbols)
        for col in range(n):
            if isinstance(latin_square[row][col], str) and latin_square[row][col] in temp_set:
                temp_set.remove(latin_square[row][col])

        for col in range(n):
            if latin_square[row][col] == "_":
                latin_square[row][col] = set(temp_set)

    # Creating a set for each column to eliminate all other symbols in sets that are already filled in each column
    for col in range(n):
        col_set = set()

        for row in range(n):
            if isinstance(latin_square[row][col], str):
                col_set.add(latin_square[row][col])

        for row in range(n):
            if isinstance(latin_square[row][col], set):
                latin_square[row][col] -= col_set

    return latin_square


def easy_solve(square):
    latin_square = square

    n = len(latin_square)
    changed = True
    while changed:
        changed = False
        for row in range(n):
            for col in range(n):
                if isinstance(latin_square[row][col], set) and len(latin_square[row][col]) == 1:
                    value = latin_square[row][col].pop()
                    latin_square[row][col] = value
                    changed = True

                    for c in range(n):
                        if isinstance(latin_square[row][c], set):
                            latin_square[row][c].discard(value)
                    for r in range(n):
                        if isinstance(latin_square[r][col], set):
                            latin_square[r][col].discard(value)

    return latin_square


def recursive_solve(square):
    latin_square = easy_solve(copy.deepcopy(square))

    n = len(latin_square)
    solved = all(isinstance(latin_square[row][col], str) for row in range(n) for col in range(n))
    if solved:
        return latin_square

    min_size = n + 1
    guess_row, guess_col = -1, -1
    for row in range(n):
        for col in range(n):
            if isinstance(latin_square[row][col], set) and len(latin_square[row][col]) < min_size:
                min_size = len(latin_square[row][col])
                guess_row, guess_col = row, col

    if guess_row == -1:
        return None

    for option in latin_square[guess_row][guess_col]:
        new_square = copy.deepcopy(latin_square)
        new_square[guess_row][guess_col] = option

        for c in range(n):
            if isinstance(new_square[guess_row][c], set):
                new_square[guess_row][c].discard(option)
        for r in range(n):
            if isinstance(new_square[r][guess_col], set):
                new_square[r][guess_col].discard(option)

        attempt = recursive_solve(new_square)
        if attempt is not None:
            return attempt

    return None

def validate_latin_square(square):
    n = len(square)
    all_symbols = set(square[0])

    for row in range(n):
        row_symbols = set(square[row])
        if row_symbols != all_symbols:
            return False

    for col in range(n):
        col_symbols = set(square[row][col] for row in range(n))
        if col_symbols != all_symbols:
            return False

    return True




def getTestCase(testfile, forexact = False):    
    with open(testfile,"r") as f:
        count = 0
        reader = csv.reader(f)
        square = []
        for row in reader:
            square.append([])
            for i in row:
                if(forexact):
                    square[count].append(str(i))
                else:
                    square[count].append(int(i))
            count+=1
        if(forexact):
            for i in range(len(square)):
                for x in range(len(square)):
                    if(square[i][x] == "0"):
                        square[i][x] = "_"
                        
        return square
    return []

def compareSquares(afterSquare,beforeSquare): #pass the changed latin square and the original to verify that it is a latinsquare and the partial square known locations weren't changed
    matched = True
    lengthFix = len(afterSquare)
    padding = 1
    print("Before:")
    while(lengthFix >= 10):
        lengthFix = lengthFix %10
        padding+= 1
    for i in beforeSquare:
        for j in i:
            print(end= "[")
            print(str(j).center(padding," "), end= "]")
        print()
    print("After:")
    for i in afterSquare:
        for j in i:
            print(end= "[")
            print(str(j).center(padding," "), end= "]")
        print()
    n = len(afterSquare)
    for row in range(n):
        row_sym = [i for i in afterSquare[row] if i != 0]
        if len(row_sym) != len(set(row_sym)):
            matched = False
        if row_sym and (min(row_sym) < 1 or max(row_sym) > n):
            matched = False
    for col in range(n):
        col_sym = [afterSquare[row][col] for row in range(n) if afterSquare[row][col] != 0]
        if len(col_sym) != len(col_sym):
            matched = False
        if col_sym and (min(col_sym) < 1 or max(col_sym) > n):
            matched = False
    print("Valid latin square (currently):",matched)
    count = 0
    for i in range(0,len(afterSquare)):
        for j in range(0,len(afterSquare)):
            if(str(afterSquare[i][j]) != str(beforeSquare[i][j]) and beforeSquare[i][j] != 0):
                raise ValueError("Square1 has changed part of the provided partially complete")
            if(afterSquare[i][j] != 0):
                count+=1
    print("Filled percent:",int(count/len(afterSquare)**2*100), end="%\n")
    print()
    return count/len(afterSquare)**2*100

#optimizing based on filled cells. Always returns a valid partial or complete square. I think this should be better but I'm honestly not smart enough to figure it out completely I can't believe it's even working 
def LPRoundingApproximation(initial_grid):
    n = len(initial_grid) 
    grid = np.full((n, n), 0)
    for i in range(n):
        for j in range(n):
            val = initial_grid[i][j]
            if val != '.':
                grid[i,j] = int(val)
    var_index = {}
    idx = 0
    for i in range(n):
        for j in range(n):
            for k in range(1, n+1):
                var_index[(i,j,k)] = idx
                idx += 1
    A_eq = []
    b_eq = []
    for i in range(n):
        for j in range(n):
            row = np.zeros(n**3)
            if grid[i,j] == 0:
                for k in range(1, n+1):
                    row[var_index[(i,j,k)]] = 1
            else:
                row[var_index[(i,j,grid[i,j])]] = 1
            A_eq.append(row)
            b_eq.append(1)
    for i in range(n):
        for k in range(1, n+1):
            row = np.zeros(n**3)
            count = 0
            for j in range(n):
                if grid[i,j] == k:
                    count += 1
                elif grid[i,j] == 0:
                    row[var_index[(i,j,k)]] = 1
            if count == 0:
                A_eq.append(row)
                b_eq.append(1)
    for j in range(n):
        for k in range(1, n+1):
            row = np.zeros(n**3)
            count = 0
            for i in range(n):
                if grid[i,j] == k:
                    count += 1
                elif grid[i,j] == 0:
                    row[var_index[(i,j,k)]] = 1
            if count == 0:
                A_eq.append(row)
                b_eq.append(1)
        c = np.zeros(n**3)
        for (i,j,k), idx in var_index.items():
            if grid[i,j] == 0:
                row_missing = n - len(set(grid[i,:])) - 1
                col_missing = n - len(set(grid[:,j])) - 1
                c[idx] = row_missing + col_missing   
    num_vars = n**3
    num_constraints = len(b_eq)
    tableau = np.zeros((num_constraints+1, num_vars + num_constraints + 1))
    tableau[:-1, :num_vars] = A_eq
    tableau[:-1, num_vars:num_vars+num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b_eq
    tableau[-1, :num_vars] = c
    basis = list(range(num_vars, num_vars + num_constraints))
    for _ in range(5000):
        entering = np.argmin(tableau[-1, :-1])
        if tableau[-1, entering] >= -1e-8:
            break
        ratios = np.inf * np.ones(num_constraints)
        for i in range(num_constraints):
            if tableau[i, entering] > 1e-8:
                ratios[i] = tableau[i, -1] / tableau[i, entering]
        leaving = np.argmin(ratios)
        if ratios[leaving] == np.inf:
            raise ValueError("Problem is unbounded - check constraints")
        basis[leaving] = entering
        pivot_val = tableau[leaving, entering]
        tableau[leaving] /= pivot_val
        for i in range(num_constraints+1):
            if i != leaving:
                tableau[i] -= tableau[i, entering] * tableau[leaving]
    solution = np.zeros(n**3)
    for i in range(num_constraints):
        if basis[i] < n**3:
            solution[basis[i]] = tableau[i, -1]
    empty_cells = [(i,j) for i in range(n) for j in range(n) if grid[i,j] == 0]
    best_result = np.copy(grid)
    best_empty = n*n 
    for attempt in range(10):
        temp = np.copy(grid)
        np.random.shuffle(empty_cells)
        for i, j in empty_cells:
            valid_ks = []
            weights = []
            for k in range(1, n+1):
                if k not in temp[i,:] and k not in temp[:,j]:
                    idx = var_index[(i,j,k)]
                    valid_ks.append(k)
                    weights.append(max(solution[idx], 1e-6))
            if valid_ks:
                weights = np.exp(weights) / np.sum(np.exp(weights))
                chosen_k = np.random.choice(valid_ks, p=weights)
                temp[i,j] = chosen_k
        current_empty = np.sum(temp == 0)
        if current_empty == 0:
            return temp.tolist()
        if current_empty < best_empty:
            best_result = temp
            best_empty = current_empty
    for i in range(n):
        for j in range(n):
            if best_result[i,j] == 0:
                available = set(range(1, n+1)) - set(best_result[i,:]) - set(best_result[:,j])
                if available:
                    symbol_counts = {k: np.sum(best_result == k) for k in available}
                    best_result[i,j] = min(symbol_counts.items(), key=lambda x: x[1])[0]

    return best_result.tolist()

def testLP(filename):
    test= getTestCase(filename)
    runningTime = 0
    filledPercent = 0
    for i in range(5):
        temp = copy.deepcopy(test)
        startTime = time.perf_counter()
        temp = LPRoundingApproximation(temp)
        stopTime = time.perf_counter()
        runningTime+= stopTime-startTime
        filledPercent += compareSquares(temp,test)
    filledPercent = filledPercent/5
    
    return int(runningTime/5*1000), int(filledPercent)
def showSquare(square):
    if square is None:
        print("There is nothing")
    else:
        for row in range(len(square)):
            print(square[row])
def testExact(filename):
    test= getTestCase(filename,True)
    runningTime = 0
    for i in range(5):
        temp = copy.deepcopy(test)
        startTime = time.perf_counter()
        the_latin_square = setup_square(temp)
        the_latin_square = easy_solve(the_latin_square)
        the_latin_square = recursive_solve(the_latin_square)
        stopTime = time.perf_counter()
        runningTime+= stopTime-startTime
        showSquare(the_latin_square)
    return int(runningTime/5*1000)
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

                    
    # all_symbols = {"1", "2", "3", "4"}
    #
    # latin_square = [
    #     ["4", "_", "1", "3"],
    #     ["1", "3", "_", "2"],
    #     ["_", "4", "_", "1"],
    #     ["2", "1", "_", "_"]
    # ]
    #
    # the_latin_square = setup_square(latin_square)
    # the_latin_square = easy_solve(the_latin_square)
    #
    # if the_latin_square is None:
    #     print("There is nothing")
    # else:
    #     for row in range(len(the_latin_square)):
    #         print(the_latin_square[row])

    


    
    time_exact_ms = testExact("dataset1_5x5.csv")
    time_exact_ms2 = testExact("dataset2_10x10.csv")
    time_exact_ms3 = testExact("dataset3_15x15.csv")
    time_ms,filled_percent = testLP("dataset1_5x5.csv")
    time_ms2,filled_percent2 = testLP("dataset2_10x10.csv")
    time_ms3,filled_percent3 = testLP("dataset3_15x15.csv")
    print("\nOn 5x5 the average running time for Exact was:",time_exact_ms,"ms")
    print("\nOn 10x10 the average running time for Exact  was:",time_exact_ms2,"ms")
    print("\nOn 15x15 the average running time for Exact  was:",time_exact_ms3,"ms")
    print("\nOn 5x5 the average filled percent was: ", filled_percent, end = "%\n")
    print("On 5x5 the average running time for LP Approximation was:",int(time_ms),"ms")
    print("\nOn 10x10 the average filled percent was: ", filled_percent2, end = "%\n")
    print("On 10x10 the average running time for LP Approximation was:",int(time_ms2),"ms")
    print("\nOn 15x15 the average filled percent was: ", filled_percent3, end = "%\n")
    print("On 15x15 the average running time for LP Approximation was:",int(time_ms3),"ms")

    #print("The average running time for Exact was",time_exact_ms,"ms")
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



