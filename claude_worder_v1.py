def load_dictionary(min_length=2):
    paths = [
        "/usr/share/dict/words",
        "/usr/dict/words",
        "words.txt"
    ]

    for path in paths:
        try:
            with open(path, "r") as f:
                print(f"Using dictionary: {path}")
                words = set()
                prefixes = set()

                for line in f:
                    word = line.strip().upper()
                    if word.isalpha() and len(word) >= min_length:
                        words.add(word)
                        for i in range(1, len(word)):
                            prefixes.add(word[:i])

                return words, prefixes
        except FileNotFoundError:
            continue

    print("Error: No dictionary found.")
    exit()


def read_grid(filename):
    with open(filename, 'r') as file:
        grid = [line.strip().upper() for line in file if line.strip()]

    size = len(grid)

    for row in grid:
        if len(row) != size:
            raise ValueError("Grid must be square")

    return grid


def parse_pattern(pattern_str):
    return [len(p) for p in pattern_str.strip().split()]


def get_neighbors(size, row, col):
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < size and 0 <= c < size:
            yield r, c


def dfs_collect(grid, row, col, visited, word, path, words, prefixes, results, max_len):
    word += grid[row][col]
    path = path + [(row, col)]
    visited.add((row, col))

    if len(word) > max_len:
        visited.remove((row, col))
        return

    if word in words:
        results.setdefault(len(word), []).append((word, set(path)))

    if word not in prefixes:
        visited.remove((row, col))
        return

    for r, c in get_neighbors(len(grid), row, col):
        if (r, c) not in visited:
            dfs_collect(grid, r, c, visited, word, path, words, prefixes, results, max_len)

    visited.remove((row, col))


def collect_all_words(grid, words, prefixes, max_len):
    results = {}

    for r in range(len(grid)):
        for c in range(len(grid)):
            dfs_collect(grid, r, c, set(), "", [], words, prefixes, results, max_len)

    return results


def find_word_combinations(words_by_length, pattern, total_cells):
    solutions = []

    def backtrack(index, chosen, used_cells):
        if index == len(pattern):
            if len(used_cells) == total_cells:
                solutions.append(chosen[:])
            return

        length_needed = pattern[index]

        if length_needed not in words_by_length:
            return

        for word, cells in words_by_length[length_needed]:
            if used_cells & cells:
                continue

            chosen.append((word, cells))
            backtrack(index + 1, chosen, used_cells | cells)
            chosen.pop()

    backtrack(0, [], set())
    return solutions


def format_columns(solutions, num_cols=5, col_width=16):
    """Format solutions into fixed number of columns, return as a string."""
    lines = []
    for row_start in range(0, len(solutions), num_cols):
        group = solutions[row_start: row_start + num_cols]
        max_lines = max(len(block) for block in group)
        padded = [block + [""] * (max_lines - len(block)) for block in group]
        for line_idx in range(max_lines):
            row = ""
            for block in padded:
                row += block[line_idx].ljust(col_width)
            lines.append(row.rstrip())
        lines.append("")  # Blank line between rows of solutions
    return "\n".join(lines)


if __name__ == "__main__":
    grid_file = input("Enter grid filename: ")
    pattern_str = input("Enter pattern (e.g. --- --- ---): ")

    grid = read_grid(grid_file)
    words, prefixes = load_dictionary()

    pattern = parse_pattern(pattern_str)
    total_cells = len(grid) * len(grid)

    if sum(pattern) != total_cells:
        print("Error: Pattern does not match total grid size.")
        exit()

    max_len = max(pattern)

    print("\nCollecting words...")
    words_by_length = collect_all_words(grid, words, prefixes, max_len)

    print("\nSearching for solutions...")
    solutions = find_word_combinations(words_by_length, pattern, total_cells)

    # Build formatted solution blocks (same structure file-2.py expected from response.txt)
    solution_blocks = []
    for i, sol in enumerate(solutions, 1):
        block = [f"Solution {i}:"]
        for word, _ in sol:
            block.append(f"  {word}")
        solution_blocks.append(block)

    header = f"Found {len(solution_blocks)} solution(s):\n\n"
    body = format_columns(solution_blocks, num_cols=5, col_width=16) if solution_blocks else "No solution found.\n"

    # Write solution.txt
    with open("solution.txt", "w", encoding="utf-8") as f:
        if not solution_blocks:
            f.write("No solution found.\n")
        else:
            f.write(header + body + "\n")

    print(f"\n{header}{body}")
    print("Results written to solution.txt")