def load_dictionary(min_length=2):
    """
    Load a system word list and build two data structures used to speed up
    the DFS search:

      words   – a set of every valid dictionary word (uppercase, alpha-only,
                at least `min_length` characters).  Used to test whether a
                candidate string formed on the grid is a real word.

      prefixes – a set of every proper prefix of every word in `words`.
                 For example, if "SNAKE" is in the dictionary, then "S",
                 "SN", "SNA", and "SNAK" are all added to prefixes.
                 During DFS we can prune any path whose current string is
                 not in prefixes – because no word can ever start with it,
                 so continuing in that direction is guaranteed to be futile.

    Tries several common dictionary file locations and falls back to a
    local "words.txt" if the system ones are absent.
    """
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
                    # Ignore entries with punctuation, digits, or that are
                    # shorter than the minimum requested length.
                    if word.isalpha() and len(word) >= min_length:
                        words.add(word)
                        # Add every proper prefix (all characters except the
                        # last) so the DFS can prune dead-end paths early.
                        for i in range(1, len(word)):
                            prefixes.add(word[:i])

                return words, prefixes
        except FileNotFoundError:
            continue

    print("Error: No dictionary found.")
    exit()


def read_grid(filename):
    """
    Read the letter grid from a plain-text file.

    Each non-empty line becomes one row of the grid; letters are uppercased
    so comparisons with the dictionary are case-insensitive.  The grid must
    be square (N rows x N columns), which is validated before returning.
    """
    with open(filename, 'r') as file:
        grid = [line.strip().upper() for line in file if line.strip()]

    size = len(grid)

    for row in grid:
        if len(row) != size:
            raise ValueError("Grid must be square")

    return grid


def parse_pattern(pattern_str):
    """
    Convert a human-readable word-length pattern into a list of integers.

    The user enters something like "--- ---- --" (groups of dashes separated
    by spaces).  The length of each dash-group is the required word length
    for that slot.  For example "--- ----" -> [3, 4].

    Using dash groups makes it easy to visualise the target answer layout.
    """
    return [len(p) for p in pattern_str.strip().split()]


def get_neighbors(size, row, col):
    """
    Yield the (row, col) coordinates of all grid cells that are adjacent
    to (row, col), including diagonals - i.e. up to 8 neighbours.

    Cells outside the grid boundary are silently skipped, so callers never
    receive out-of-range indices.
    """
    # All eight cardinal and diagonal directions as (row-delta, col-delta).
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
    """
    Recursive depth-first search that extends the current path by one cell
    and records any valid dictionary words found along the way.

    How it works
    ------------
    At each call we are sitting on cell (row, col).  We append its letter to
    the running `word` string and record the cell in `path`.

    Two pruning conditions allow us to abandon branches early:

    1. Length cap - if `word` is already longer than the longest word length
       required by the pattern, there is no point going deeper.

    2. Prefix check - if `word` is not a prefix of *any* dictionary word it
       can never lead to a valid word, so we stop immediately.  This is the
       key optimisation: without it the DFS would explore every permutation
       of grid cells.

    When `word` is found in the full-word set it is saved in `results`,
    keyed by its length so the combination search can look up words of a
    specific length in O(1).

    The `visited` set prevents reusing the same cell within a single word
    path.  Crucially, cells are *removed* from `visited` on the way back up
    (backtracking) so that sibling branches can reuse those cells.

    Parameters
    ----------
    grid      : list of strings - the letter grid
    row, col  : current cell coordinates
    visited   : set of (r, c) tuples already used on this path
    word      : the string built so far (not including this cell yet)
    path      : list of (r, c) tuples forming the current path
    words     : full dictionary word set
    prefixes  : set of all proper prefixes of dictionary words
    results   : dict mapping word-length -> list of (word, set-of-cells)
    max_len   : maximum word length we ever need (caps the search depth)
    """
    # Extend the current word and path with this cell.
    word += grid[row][col]
    path = path + [(row, col)]
    visited.add((row, col))

    # Pruning 1: exceeded the longest word length we could ever need.
    if len(word) > max_len:
        visited.remove((row, col))
        return

    # Record the word if it appears in the dictionary.
    if word in words:
        # Store the cell set so the combination stage can check for overlaps.
        results.setdefault(len(word), []).append((word, set(path)))

    # Pruning 2: current string is not a prefix of any dictionary word,
    # so no extension of this path can ever produce a valid word.
    if word not in prefixes:
        visited.remove((row, col))
        return

    # Recurse into every unvisited neighbour.
    for r, c in get_neighbors(len(grid), row, col):
        if (r, c) not in visited:
            dfs_collect(grid, r, c, visited, word, path, words, prefixes, results, max_len)

    # Backtrack: unmark this cell so other branches can use it.
    visited.remove((row, col))


def collect_all_words(grid, words, prefixes, max_len):
    """
    Launch a DFS from every cell in the grid and collect all valid
    dictionary words that can be traced by moving between adjacent cells
    without reusing any cell within the same word.

    Returns a dict:  { word_length: [(word, set_of_cells), ...], ... }

    Keying by length lets the combination search instantly retrieve all
    candidates of the exact length required for each pattern slot.
    """
    results = {}

    # Every cell is a potential starting point.
    for r in range(len(grid)):
        for c in range(len(grid)):
            dfs_collect(grid, r, c, set(), "", [], words, prefixes, results, max_len)

    return results


def find_word_combinations(words_by_length, pattern, total_cells):
    """
    Find every ordered combination of words whose lengths match the pattern
    and whose grid cells together cover every cell in the grid exactly once
    (no cell used by two words, no cell left unused).

    Algorithm - recursive backtracking
    ------------------------------------
    We fill pattern slots one by one (left to right).  For each slot we try
    every candidate word of the required length.  Before committing to a
    word we check that none of its cells overlap with cells already claimed
    by previously chosen words (using a set intersection: `used_cells & cells`).

    If a word passes the overlap check we:
      1. Add it to `chosen` and merge its cells into `used_cells`.
      2. Recurse to fill the next slot.
      3. Pop it back off (`chosen.pop()`) to try the next candidate -
         this is the backtracking step that undoes the choice.

    When all slots are filled (`index == len(pattern)`) we confirm that
    the union of chosen cells equals the full grid (len(used_cells) ==
    total_cells).  Only then is the combination recorded as a solution.

    Parameters
    ----------
    words_by_length : dict of { length: [(word, cell_set), ...] }
    pattern         : list of required word lengths, e.g. [3, 4, 5]
    total_cells     : total number of cells in the grid (must all be covered)
    """
    solutions = []

    def backtrack(index, chosen, used_cells):
        # Base case: all pattern slots have been filled.
        if index == len(pattern):
            # Accept only if every grid cell has been used exactly once.
            if len(used_cells) == total_cells:
                solutions.append(chosen[:])
            return

        length_needed = pattern[index]

        # No words of this length were found on the grid - dead end.
        if length_needed not in words_by_length:
            return

        for word, cells in words_by_length[length_needed]:
            # Skip this word if any of its cells are already taken.
            if used_cells & cells:
                continue

            # Commit: add the word and claim its cells.
            chosen.append((word, cells))
            backtrack(index + 1, chosen, used_cells | cells)
            # Backtrack: undo the choice so we can try the next candidate.
            chosen.pop()

    backtrack(0, [], set())
    return solutions


def format_columns(solutions, num_cols=5, col_width=16):
    """
    Lay out solution blocks side-by-side in a fixed-column grid for compact,
    readable output.

    Each element of `solutions` is a list of strings (the header line plus
    one line per word).  Blocks are arranged left-to-right, `num_cols` per
    row.  Within a row, shorter blocks are padded with blank lines so all
    blocks in the row share the same height, enabling clean column alignment
    via ljust().

    A blank line is inserted between each row of blocks to visually separate
    them.
    """
    lines = []
    for row_start in range(0, len(solutions), num_cols):
        # Slice out the next group of up to num_cols blocks.
        group = solutions[row_start: row_start + num_cols]

        # Determine the tallest block in this group so shorter ones can be
        # padded to the same height.
        max_lines = max(len(block) for block in group)
        padded = [block + [""] * (max_lines - len(block)) for block in group]

        # Interleave one line from each block, left-justified to col_width.
        for line_idx in range(max_lines):
            row = ""
            for block in padded:
                row += block[line_idx].ljust(col_width)
            lines.append(row.rstrip())

        lines.append("")  # Blank separator between rows of solution blocks.

    return "\n".join(lines)


def print_unique_words(solutions, words_per_line=8):
    """
    Collect every unique word that appears across all solutions and print
    them to the screen, sorted alphabetically, with at most `words_per_line`
    words on each line.

    Words are gathered directly from the in-memory solution list rather than
    re-reading solution.txt, mirroring the intent of main.py (which extracted
    indented words from the file) but without the extra I/O round-trip.
    """
    # Flatten all (word, cells) pairs from every solution and keep only the
    # word strings, then deduplicate with a set.
    unique_words = sorted({word for sol in solutions for word, _ in sol})

    print(f"\nUnique words across all solutions ({len(unique_words)} total):")

    # Print in rows of `words_per_line`, padding each word to a fixed width
    # so columns stay aligned regardless of word length.
    col_width = max((len(w) for w in unique_words), default=0) + 2
    for i in range(0, len(unique_words), words_per_line):
        row = unique_words[i: i + words_per_line]
        print("  " + "".join(w.ljust(col_width) for w in row))


if __name__ == "__main__":
    grid_file = input("Enter grid filename: ")
    pattern_str = input("Enter pattern (e.g. --- --- ---): ")

    grid = read_grid(grid_file)
    words, prefixes = load_dictionary()

    pattern = parse_pattern(pattern_str)
    total_cells = len(grid) * len(grid)

    # The pattern must account for every cell - no gaps, no overlaps.
    if sum(pattern) != total_cells:
        print("Error: Pattern does not match total grid size.")
        exit()

    # We only need words up to the length of the longest pattern slot.
    max_len = max(pattern)

    print("\nCollecting words...")
    words_by_length = collect_all_words(grid, words, prefixes, max_len)

    print("\nSearching for solutions...")
    solutions = find_word_combinations(words_by_length, pattern, total_cells)

    # Convert the raw solutions (list of (word, cell_set) tuples) into
    # pre-formatted text blocks ready for display and file output.
    solution_blocks = []
    for i, sol in enumerate(solutions, 1):
        block = [f"Solution {i}:"]
        for word, _ in sol:
            block.append(f"  {word}")
        solution_blocks.append(block)

    header = f"Found {len(solution_blocks)} solution(s):\n\n"
    body = format_columns(solution_blocks, num_cols=5, col_width=16) if solution_blocks else "No solution found.\n"

    # Write the final formatted output to solution.txt.
    with open("solution.txt", "w", encoding="utf-8") as f:
        if not solution_blocks:
            f.write("No solution found.\n")
        else:
            f.write(header + body + "\n")

    print(f"\n{header}{body}")
    print("Results written to solution.txt")

    # Display the deduplicated word list, 4 per line.
    print_unique_words(solutions)