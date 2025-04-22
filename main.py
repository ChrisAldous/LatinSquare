import copy
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
