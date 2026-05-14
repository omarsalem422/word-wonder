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


def find_word_path(grid, word, excluded_cells):
    """
    Find a single valid path for `word` on the grid that does not use any
    cell in `excluded_cells` (cells already claimed by previously verified
    words).

    Uses a simple DFS from every eligible starting cell.  Returns a set of
    (row, col) tuples for the path, or None if the word cannot be walked.

    This is used in verify_known_words to confirm user-supplied answers.
    """
    size = len(grid)

    def dfs(r, c, idx, visited):
        if idx == len(word):
            return list(visited)
        for nr, nc in get_neighbors(size, r, c):
            if (nr, nc) not in visited and (nr, nc) not in excluded_cells \
                    and grid[nr][nc] == word[idx]:
                visited.append((nr, nc))
                result = dfs(nr, nc, idx + 1, visited)
                if result is not None:
                    return result
                visited.pop()
        return None

    for r in range(size):
        for c in range(size):
            if (r, c) not in excluded_cells and grid[r][c] == word[0]:
                path = dfs(r, c, 1, [(r, c)])
                if path:
                    return set(path)
    return None


def verify_known_words(grid, known_words):
    """
    Verify that a list of user-supplied words can all be walked on the grid
    with no cell reused and every cell covered exactly once.

    For each word in order we attempt to find a valid path through the cells
    not yet claimed by previous words.  If any word fails, or if the union
    of all paths does not cover every cell, verification fails.

    Returns (solutions, error_message) where solutions is a single-element
    list in the same format as find_word_combinations (so the rest of the
    program needs no changes), or an empty list on failure.
    """
    total_cells = len(grid) * len(grid)
    used_cells = set()
    solution = []

    for word in known_words:
        path = find_word_path(grid, word.upper(), used_cells)
        if path is None:
            return [], f"'{word}' cannot be walked on the grid without reusing cells."
        solution.append((word.upper(), path))
        used_cells |= path

    if len(used_cells) != total_cells:
        uncovered = total_cells - len(used_cells)
        return [], f"Words verified individually but {uncovered} cell(s) left uncovered."

    return [solution], None  # wrap in list to match find_word_combinations format


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


def load_valid_english_words(word_list_path="words.txt"):
    """
    Load an English word list into a set for O(1) membership checks.

    Each line is stripped of whitespace and uppercased so comparisons are
    case-insensitive and consistent with how solution words are stored.
    Returns an empty set (with a warning) if the file cannot be found, so
    the rest of the program can continue gracefully.
    """
    try:
        with open(word_list_path, "r") as f:
            return {line.strip().upper() for line in f if line.strip()}
    except FileNotFoundError:
        print(f"Warning: word list '{word_list_path}' not found; English filter unavailable.")
        return set()


def print_word_list(label, word_list, words_per_line=4):
    """
    Print a sorted list of words under a labelled heading, with at most
    `words_per_line` words per line and consistent column alignment.
    """
    if not word_list:
        print(f"\n{label} (0 total): none")
        return

    print(f"\n{label} ({len(word_list)} total):")

    # Pad every word to the same width so columns line up neatly.
    col_width = max(len(w) for w in word_list) + 2
    for i in range(0, len(word_list), words_per_line):
        row = word_list[i: i + words_per_line]
        print("  " + "".join(w.ljust(col_width) for w in row))


def print_unique_words(solutions, valid_english_words, words_per_line=4):
    """
    Display two word lists derived from all solutions:

    1. Unique words  – every distinct word that appears in any solution,
                       sorted alphabetically.

    2. English words – the subset of unique words confirmed to exist in the
                       English word list (loaded via load_valid_english_words).
                       This filters out grid paths that happen to match the
                       dictionary used for the puzzle search but are not
                       recognisable English words (e.g. obscure abbreviations
                       or proper nouns that sneak into system dictionaries).

    Words are gathered directly from the in-memory solution list rather than
    re-reading solution.txt, avoiding an unnecessary file I/O round-trip.

    Parameters
    ----------
    solutions           : raw output of find_word_combinations –
                          list of [(word, cell_set), ...] lists
    valid_english_words : set of uppercase English words from load_valid_english_words
    words_per_line      : how many words to print per line (default 4)
    """
    # Flatten all (word, cells) pairs across every solution, keep only the
    # word strings, then deduplicate and sort.
    unique_words = sorted({word for sol in solutions for word, _ in sol})

    # Filter to words confirmed present in the English word list.
    english_words = [w for w in unique_words if w in valid_english_words]

    print_word_list("Unique words across all solutions", unique_words, words_per_line)
    print_word_list("English words (filtered)", english_words, words_per_line)


def format_word_list_text(label, word_list, words_per_line=4):
    """
    Return a formatted string for a word list suitable for writing to a file.
    Layout mirrors print_word_list() but produces a string instead of printing.
    """
    if not word_list:
        return f"{label} (0 total): none\n"

    col_width = max(len(w) for w in word_list) + 2
    lines = [f"{label} ({len(word_list)} total):"]
    for i in range(0, len(word_list), words_per_line):
        row = word_list[i: i + words_per_line]
        lines.append("  " + "".join(w.ljust(col_width) for w in row))
    return "\n".join(lines) + "\n"


def build_solution_file_content(solutions, valid_english_words, grid, grid_file, pattern_str,
                                solution_blocks, body, free_search_used=False, words_per_line=4):
    """
    Assemble the full text content for solution.txt in the required order:

      1. Unique words found across all solutions
      2. English-only subset of those unique words
      3. The raw grid (content of grid.txt)
      4. The pattern the user entered
      5. The formatted solutions (existing output)

    Returns the complete string ready to be written to disk.
    """
    # --- Section 1 & 2: unique words and English-filtered words ---
    unique_words = sorted({word for sol in solutions for word, _ in sol})
    english_words = [w for w in unique_words if w in valid_english_words]

    section_unique  = format_word_list_text("Unique words across all solutions", unique_words, words_per_line)
    section_english = format_word_list_text("English words (filtered)", english_words, words_per_line)

    # --- Section 3: grid contents ---
    grid_lines = "\n".join(grid)
    section_grid = f"Grid ({grid_file}):\n{grid_lines}\n"

    # --- Section 4: pattern ---
    search_mode = "user-verified (proper nouns / non-dictionary words)" if free_search_used else "dictionary search"
    section_pattern = f"Pattern: {pattern_str.strip()}\nSearch mode: {search_mode}\n"

    # --- Section 5: solutions ---
    if not solution_blocks:
        section_solutions = "No solution found.\n"
    else:
        header = f"Found {len(solution_blocks)} solution(s):\n\n"
        section_solutions = header + body + "\n"

    separator = "\n" + "-" * 60 + "\n"
    return (
        section_unique
        + separator
        + section_english
        + separator
        + section_grid
        + separator
        + section_pattern
        + separator
        + section_solutions
    )


def save_execution_history(content, history_dir="Execution-History"):
    """
    Save `content` to the Execution-History folder using a timestamp filename.

    The folder is created automatically if it does not already exist.
    The filename format is MM-DD-YYYY-HHMMSS.txt so that files sort
    chronologically when listed by name.
    """
    import os
    from datetime import datetime

    os.makedirs(history_dir, exist_ok=True)

    # Format: month-day-year-time  e.g. 05-10-2026-143022.txt
    timestamp = datetime.now().strftime("%m-%d-%Y-%H%M%S")
    history_path = os.path.join(history_dir, f"{timestamp}.txt")

    with open(history_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Execution history saved to: {history_path}")


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

    print("\nCollecting words (dictionary search)...")
    words_by_length = collect_all_words(grid, words, prefixes, max_len)

    print("Searching for solutions...")
    solutions = find_word_combinations(words_by_length, pattern, total_cells)

    # If the dictionary search found nothing, the answer likely contains
    # proper nouns or brand names not in the dictionary.
    # Prompt the user to enter their known words and verify them on the grid.
    free_search_used = False
    if not solutions:
        print("\nNo dictionary solutions found.")
        print("This can happen when answers are proper nouns or brand names.")
        print("If you know the answer, enter the words now to verify them.")
        print("(Press Enter with no input to skip.)")

        known_input = input("Enter your known words separated by spaces: ").strip()
        if known_input:
            known_words = known_input.split()
            # Check word lengths match the pattern
            known_lengths = [len(w) for w in known_words]
            if sorted(known_lengths) != sorted(pattern):
                print(f"Error: word lengths {known_lengths} do not match "
                      f"pattern {pattern}. Skipping verification.")
            else:
                solutions, error = verify_known_words(grid, known_words)
                free_search_used = True
                if solutions:
                    print(f"Verified! All words walk the grid correctly and "
                          f"cover all {len(grid)*len(grid)} cells.")
                else:
                    print(f"Verification failed: {error}")

    # Convert the raw solutions (list of (word, cell_set) tuples) into
    # pre-formatted text blocks ready for display and file output.
    solution_blocks = []
    for i, sol in enumerate(solutions, 1):
        block = [f"Solution {i}:"]
        for word, _ in sol:
            block.append(f"  {word}")
        solution_blocks.append(block)

    body = format_columns(solution_blocks, num_cols=5, col_width=16) if solution_blocks else ""

    # Load the English word list for filtering unique words.
    valid_english_words = load_valid_english_words("words.txt")

    # Build the full structured file content in the required order:
    #   1. Unique words  2. English words  3. Grid  4. Pattern  5. Solutions
    file_content = build_solution_file_content(
        solutions, valid_english_words, grid, grid_file,
        pattern_str, solution_blocks, body, free_search_used
    )

    # Write solution.txt.
    with open("solution.txt", "w", encoding="utf-8") as f:
        f.write(file_content)
    print("Results written to solution.txt")

    # Save a timestamped copy to Execution-History/.
    save_execution_history(file_content)

    # Print the word lists and solutions to the screen as before.
    print_unique_words(solutions, valid_english_words)
    if solution_blocks:
        print(f"\nFound {len(solution_blocks)} solution(s):\n\n{body}")
    else:
        print("\nNo solution found.")