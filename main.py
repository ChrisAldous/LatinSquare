import copy
import csv
import random
import math
import numpy as np
import time

def setup_square(square):
    latin_square = square

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

                    # After placing, eliminate from row and column
                    for c in range(n):
                        if isinstance(latin_square[row][c], set):
                            latin_square[row][c].discard(value)
                    for r in range(n):
                        if isinstance(latin_square[r][col], set):
                            latin_square[r][col].discard(value)

    return recursive_solve(latin_square)


def recursive_solve(square):
    latin_square = square
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

    for option in latin_square[guess_row][guess_col]:
        new_square = copy.deepcopy(latin_square)
        new_square[guess_row][guess_col] = option

        for c in range(n):
            if isinstance(new_square[guess_row][c], set):
                new_square[guess_row][c].discard(option)
        for r in range(n):
            if isinstance(new_square[r][guess_col], set):
                new_square[r][guess_col].discard(option)

        new_square = easy_solve(new_square)

        result = recursive_solve(new_square)
        if result is not None:
            return result

    return None

def validate_latin_square(square):
    n = len(square)
    all_symbols = set(square[0])
    
    for row in range(n):
        row_sym = set(square[row])
        if row_sym != all_symbols:
            return False

    for col in range(n):
        col_sym = set(square[row][col] for row in range(n))
        if col_sym != all_symbols:
            return False

    return True


def getTestCase(num = 0):
    if(num < 1 or num > 30):
        num = random.randrange(1,28)
    print("Getting id number:",num)
    with open("dataset.csv","r") as f:
        count = 0
        reader = csv.reader(f)
        for row in reader:
            if(count == num):
                square = []
                solution_square = []
                tempcount = 0
                tempcount2 = 0
                square_size = int(math.sqrt(len(row[1])))
                for i in range(square_size):
                    square.append([])
                    solution_square.append([])
                for i in row[1]:
                        if(i == '.'):
                            i =0
                        i =int(i)
                        square[tempcount].append(i)
                        tempcount2+=1
                        if(tempcount2 >= square_size):
                            tempcount2 = 0
                            tempcount += 1
                tempcount = 0
                tempcount2 = 0
                for i in row[2]:
                        if(i == '.'):
                            i =0
                        i =int(i)
                        solution_square[tempcount].append(i)
                        tempcount2+=1
                        if(tempcount2 >= square_size):
                            tempcount2 = 0
                            tempcount += 1
                return square,solution_square
            count+=1
    return [], []

def compareSquares(square1,square2): #pass the changed latin square and the original to verify that it is a latinsquare and the partial square known locations weren't changed
    matched = True
    print("Before:")
    for i in square2:
        for j in i:
            print(end= "[")
            print(j, end ="]")
        print()
    print("After:")
    for i in square1:
        for j in i:
            print(end= "[")
            print(j, end ="]")
        print()
    n = len(square1)
    for row in range(n):
        row_sym = [i for i in square1[row] if i != 0]
        if len(row_sym) != len(set(row_sym)):
            matched = False
        if row_sym and (min(row_sym) < 1 or max(row_sym) > n):
            matched = False
    for col in range(n):
        col_sym = [square1[row][col] for row in range(n) if square1[row][col] != 0]
        if len(col_sym) != len(col_sym):
            matched = False
        if col_sym and (min(col_sym) < 1 or max(col_sym) > n):
            matched = False
    print("Valid latin square (so far):",matched)
    count = 0
    for i in range(0,len(square1)):
        for j in range(0,len(square1)):
            if(str(square1[i][j]) != str(square2[i][j]) and square2[i][j] != 0):
                raise ValueError("Square1 has changed part of the provided partially complete")
            if(square1[i][j] != 0):
                count+=1
    print("Filled percent:",count/len(square1)**2*100)
    print()
    return count/len(square1)**2*100

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

def testLP():
    print("Starting LP approximation test")
    test, test_solution = getTestCase(10)
    test2,test_solution2 = getTestCase(28)
    test3,test_solution3 = getTestCase(29)
    runningTime = 0
    filledPercent = 0
    for i in range(5):
        startTime = time.perf_counter()
        temp = LPRoundingApproximation(test)
        stopTime = time.perf_counter()
        runningTime+= stopTime-startTime
        filledPercent += compareSquares(temp,test)
        startTime = time.perf_counter()
        temp = LPRoundingApproximation(test2)
        stopTime = time.perf_counter()
        runningTime+= stopTime-startTime
        filledPercent +=compareSquares(temp,test2)
        startTime = time.perf_counter()
        temp = LPRoundingApproximation(test3)
        stopTime = time.perf_counter()
        runningTime+= stopTime-startTime
        filledPercent += compareSquares(temp,test3)
        
        
    print("Ending LP approximation test")
    filledPercent = filledPercent/15
    print("The average filled percent was: ", int(filledPercent), end = "%\n")
    return runningTime/15*1000
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


    all_symbols = {"1", "2", "3", "4", "5"}

    latin_square = [
        ["4", "_", "_", "3", "5"],
        ["1", "3", "_", "5", "2"],
        ["_", "4", "_", "1", "_"],
        ["_", "_", "1", "_", "_"],
        ["4", "_", "_", "3", "1"]
    ]
    the_latin_square = setup_square(latin_square)
    the_latin_square = easy_solve(the_latin_square)
    if the_latin_square is None:
        print("There is nothing")
    else:
        for row in range(len(the_latin_square)):
            print(the_latin_square[row])

    
    print("The average running time for LP Approximation was:",int(testLP()),"ms")
