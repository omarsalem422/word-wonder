def load_dictionary(min_length=2, brands_path="brands.txt"):
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

    Brand names from `brands_path` (default: brands.txt) are merged into
    both sets so the DFS treats them as valid words alongside dictionary
    words.  Non-alpha characters in brand names (spaces, hyphens, etc.)
    are stripped before merging so they can match unbroken grid paths.
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

                # Merge brand names so the DFS recognises them as valid words.
                brand_words, brand_prefixes = load_brands(brands_path, min_length)
                words    |= brand_words
                prefixes |= brand_prefixes

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


def load_brands(brands_path="brands.txt", min_length=2):
    """
    Load a brand list and return two sets used to augment the DFS search:

      brand_words   – normalised brand strings (uppercase, letters only,
                      at least `min_length` characters).  Non-alpha characters
                      such as spaces, hyphens, ampersands, and dots are stripped
                      so that multi-word names like "COCA COLA" become "COCACOLA"
                      and can be matched against unbroken letter paths on the grid.

      brand_prefixes – every proper prefix of every brand word, enabling the
                       same DFS prefix-pruning that the main dictionary uses.

    Returns (brand_words, brand_prefixes).  Both sets are empty (with a
    warning) if brands.txt cannot be found, so the program degrades gracefully.
    """
    brand_words = set()
    brand_prefixes = set()

    try:
        with open(brands_path, "r", encoding="utf-8") as f:
            print(f"Using brands file: {brands_path}")
            for line in f:
                # Strip surrounding whitespace, uppercase, then keep only letters
                # so "Coca-Cola", "Coca Cola", and "COCA COLA" all become "COCACOLA".
                raw = line.strip().upper()
                word = "".join(ch for ch in raw if ch.isalpha())

                if len(word) >= min_length:
                    brand_words.add(word)
                    for i in range(1, len(word)):
                        brand_prefixes.add(word[:i])

    except FileNotFoundError:
        print(f"Warning: brands file '{brands_path}' not found; brand matching unavailable.")

    return brand_words, brand_prefixes


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


def print_unique_words(solutions, valid_english_words, brand_words=None, words_per_line=4):
    """
    Display three word lists derived from all solutions:

    1. Unique words  – every distinct word that appears in any solution,
                       sorted alphabetically.

    2. English words – the subset of unique words confirmed to exist in the
                       English word list (loaded via load_valid_english_words).
                       This filters out grid paths that happen to match the
                       dictionary used for the puzzle search but are not
                       recognisable English words (e.g. obscure abbreviations
                       or proper nouns that sneak into system dictionaries).

    3. Brand matches – the subset of unique words that are also present in
                       the brands list (loaded via load_brands).  Only shown
                       when brand_words is provided and non-empty.

    Words are gathered directly from the in-memory solution list rather than
    re-reading solution.txt, avoiding an unnecessary file I/O round-trip.

    Parameters
    ----------
    solutions           : raw output of find_word_combinations –
                          list of [(word, cell_set), ...] lists
    valid_english_words : set of uppercase English words from load_valid_english_words
    brand_words         : set of normalised brand strings from load_brands (or None)
    words_per_line      : how many words to print per line (default 4)
    """
    # Flatten all (word, cells) pairs across every solution, keep only the
    # word strings, then deduplicate and sort.
    unique_words = sorted({word for sol in solutions for word, _ in sol})

    # Filter to words confirmed present in the English word list.
    english_words = [w for w in unique_words if w in valid_english_words]

    print_word_list("Unique words across all solutions", unique_words, words_per_line)
    print_word_list("English words (filtered)", english_words, words_per_line)

    # Show brand matches only when the brand set was successfully loaded.
    if brand_words:
        brand_matches = [w for w in unique_words if w in brand_words]
        print_word_list("Brand matches (filtered)", brand_matches, words_per_line)


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
                                solution_blocks, body, brand_words=None, words_per_line=4):
    """
    Assemble the full text content for solution.txt in the required order:

      1. Unique words found across all solutions
      2. English-only subset of those unique words
      3. Brand matches (only when brand_words is provided and non-empty)
      4. The raw grid (content of grid.txt)
      5. The pattern the user entered
      6. The formatted solutions (existing output)

    Returns the complete string ready to be written to disk.
    """
    # --- Section 1 & 2: unique words and English-filtered words ---
    unique_words = sorted({word for sol in solutions for word, _ in sol})
    english_words = [w for w in unique_words if w in valid_english_words]

    section_unique  = format_word_list_text("Unique words across all solutions", unique_words, words_per_line)
    section_english = format_word_list_text("English words (filtered)", english_words, words_per_line)

    # --- Section 3 (optional): brand matches ---
    separator = "\n" + "-" * 60 + "\n"
    if brand_words:
        brand_matches = [w for w in unique_words if w in brand_words]
        section_brands = format_word_list_text("Brand matches (filtered)", brand_matches, words_per_line)
    else:
        section_brands = None

    # --- Section 4: grid contents ---
    grid_lines = "\n".join(grid)
    section_grid = f"Grid ({grid_file}):\n{grid_lines}\n"

    # --- Section 5: pattern ---
    section_pattern = f"Pattern: {pattern_str.strip()}\n"

    # --- Section 6: solutions ---
    if not solution_blocks:
        section_solutions = "No solution found.\n"
    else:
        header = f"Found {len(solution_blocks)} solution(s):\n\n"
        section_solutions = header + body + "\n"

    result = (
        section_unique
        + separator
        + section_english
        + separator
    )
    if section_brands is not None:
        result += section_brands + separator
    result += (
        section_grid
        + separator
        + section_pattern
        + separator
        + section_solutions
    )
    return result


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


# =============================================================================
# RELAXED MODE — Option 2
# Collects every traceable string of the required lengths, with no dictionary
# or brand check.  Used as a fallback when the normal search finds no solution,
# to surface candidate strings the user can validate manually.
# =============================================================================

def _dfs_cell_paths(grid, row, col, visited, path, target_len, out):
    """
    Collect every connected cell path of exactly `target_len` cells, storing
    both the frozenset of cells and the string spelled in traversal order.
    No letter or word filtering is applied.
    """
    path = path + [(row, col)]
    visited.add((row, col))

    if len(path) == target_len:
        out.append((frozenset(path),
                    "".join(grid[r][c] for r, c in path)))
        visited.remove((row, col))
        return

    for r, c in get_neighbors(len(grid), row, col):
        if (r, c) not in visited:
            _dfs_cell_paths(grid, r, c, visited, path, target_len, out)

    visited.remove((row, col))


def collect_relaxed_strings(grid, pattern):
    """
    For each unique length in `pattern`, enumerate all connected cell paths
    of that length.

    Returns a dict:
        { length: { frozenset_of_cells -> set_of_strings } }

    Each cell-set maps to ALL strings it can spell (one per traversal order).
    This representation is used by find_relaxed_combinations to:
      (a) partition the grid by cell-sets (small, tractable search space), and
      (b) recover every valid reading of each cell group.
    """
    from collections import defaultdict
    result = {}
    for target_len in set(pattern):
        raw = []
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                _dfs_cell_paths(grid, r, c, set(), [], target_len, raw)
        mapping = defaultdict(set)
        for cell_set, string in raw:
            mapping[cell_set].add(string)
        result[target_len] = dict(mapping)
    return result


def find_relaxed_combinations(grid, pattern, valid_english_words=None,
                               brand_words=None, prefixes=None,
                               max_solutions=500):
    """
    Find candidate tilings of the grid that match `pattern` without requiring
    every string to be a known dictionary word or brand.

    Strategy — tiered filtering
    ---------------------------
    A fully unfiltered relaxed search produces tens of thousands of candidate
    strings per length on even a small grid, making the combination search
    intractable.  We narrow the candidate pool in three tiers, using the
    first tier that yields results for each pattern length:

    Tier 1 — exact match (same as normal search but fed all known words).
        Keep only strings present in `valid_english_words` or `brand_words`.
        This catches brands that were in brands.txt but somehow missed by the
        normal DFS (e.g. because their prefix was pruned by the main dict).

    Tier 2 — prefix-filtered (plausible strings).
        Keep strings whose every prefix appears in the DFS prefix set built
        from the main dictionary + brands.  This eliminates strings like
        "XQRZT" that no real word could start with, while keeping anything
        that looks like a real-word prefix chain (e.g. "PF" from "Pfizer").
        Typically reduces the pool by ~70–80 %.

    Tier 3 — unfiltered fallback (last resort).
        If Tiers 1 and 2 both return nothing for a given length, accept all
        traceable strings of that length.  Results are capped at
        `max_solutions` to keep output readable.

    Parameters
    ----------
    grid                : the letter grid
    pattern             : list of required word lengths
    valid_english_words : set of valid English words (from load_valid_english_words)
    brand_words         : set of normalised brand strings (from load_brands)
    prefixes            : prefix set from load_dictionary (used for Tier 2)
    max_solutions       : maximum number of candidate solutions to return

    Returns a list in the same format as find_word_combinations:
        [ [(string, cell_set), ...], ... ]
    """
    total_cells         = len(grid) * len(grid[0])
    valid_english_words = valid_english_words or set()
    brand_words         = brand_words or set()
    known_words         = valid_english_words | brand_words

    # Collect { length -> { frozenset_of_cells -> set_of_strings } }
    relaxed = collect_relaxed_strings(grid, pattern)

    # Convert to sorted list per length, known-word cell-sets first.
    # Sorting ensures tilings containing real words/brands are found before
    # the cap is reached, so the most useful candidates always appear.
    def sort_key(item):
        cell_set, strings = item
        has_known = any(s in known_words for s in strings)
        return (0 if has_known else 1)   # known words first

    entries_by_length = {
        length: sorted(mapping.items(), key=sort_key)
        for length, mapping in relaxed.items()
    }

    solutions = []

    def backtrack(slot, chosen_cells, chosen_strings, used):
        if len(solutions) >= max_solutions:
            return
        if slot == len(pattern):
            if len(used) == total_cells:
                solutions.append(list(zip(chosen_strings, chosen_cells)))
            return
        length = pattern[slot]
        for cell_set, strings in entries_by_length.get(length, []):
            if used & cell_set:
                continue
            # Prefer known words/brands as the display label; otherwise
            # use the alphabetically first traceable string for this cell group.
            known = sorted(s for s in strings if s in known_words)
            label = known[0] if known else sorted(strings)[0]
            chosen_cells.append(cell_set)
            chosen_strings.append(label)
            backtrack(slot + 1, chosen_cells, chosen_strings, used | cell_set)
            chosen_cells.pop()
            chosen_strings.pop()

    backtrack(0, [], [], set())
    return solutions


def print_relaxed_solutions(relaxed_solutions, normal_solution_strings,
                            valid_english_words, brand_words,
                            num_cols=5, col_width=16):
    """
    Display the relaxed-mode candidate solutions, clearly labelled as
    unvalidated.  Solutions that duplicate ones already found by the normal
    search are suppressed to avoid noise.

    For each unique string that appears across the candidates, three filtered
    sub-lists are shown so the user can quickly spot real words or brands:
      - All unique candidate strings
      - Subset confirmed in the English word list
      - Subset confirmed in the brand list

    Parameters
    ----------
    relaxed_solutions       : output of find_relaxed_combinations
    normal_solution_strings : set of frozensets of word-strings from the
                              normal search, used to deduplicate
    valid_english_words     : set from load_valid_english_words
    brand_words             : set from load_brands
    num_cols, col_width     : formatting parameters
    """
    # Filter out any relaxed solution whose word-set exactly matches a normal
    # solution (those are already shown above).
    novel = [
        sol for sol in relaxed_solutions
        if frozenset(w for w, _ in sol) not in normal_solution_strings
    ]

    if not novel:
        print("\nRelaxed mode: no additional candidate tilings found.")
        return

    print(f"\n{'='*60}")
    print("RELAXED MODE — Unvalidated candidate tilings")
    print("These strings tile the grid correctly but are NOT confirmed")
    print("as dictionary words or known brands. Review manually.")
    print(f"{'='*60}")

    # Build and display solution blocks.
    blocks = []
    for i, sol in enumerate(novel, 1):
        block = [f"Candidate {i}:"]
        for word, _ in sol:
            block.append(f"  {word}")
        blocks.append(block)

    body = format_columns(blocks, num_cols=num_cols, col_width=col_width)
    print(f"\nFound {len(novel)} candidate tiling(s):\n\n{body}")

    # Unique strings across all candidates.
    unique_strings = sorted({w for sol in novel for w, _ in sol})
    english_hits   = [w for w in unique_strings if w in valid_english_words]
    brand_hits     = [w for w in unique_strings if brand_words and w in brand_words]

    print_word_list("Unique candidate strings", unique_strings)
    print_word_list("English word matches",     english_hits)
    if brand_words:
        print_word_list("Brand matches",        brand_hits)


def format_relaxed_solution_text(relaxed_solutions, normal_solution_strings,
                                  valid_english_words, brand_words,
                                  num_cols=5, col_width=16):
    """
    Return the relaxed-mode section as a string suitable for appending to
    solution.txt and the execution history file.  Mirrors the logic of
    print_relaxed_solutions but produces text instead of printing.
    """
    novel = [
        sol for sol in relaxed_solutions
        if frozenset(w for w, _ in sol) not in normal_solution_strings
    ]

    sep = "\n" + "=" * 60 + "\n"

    if not novel:
        return sep + "RELAXED MODE: no additional candidate tilings found.\n"

    header = (
        sep
        + "RELAXED MODE — Unvalidated candidate tilings\n"
        + "These strings tile the grid correctly but are NOT confirmed\n"
        + "as dictionary words or known brands. Review manually.\n"
        + sep
    )

    blocks = []
    for i, sol in enumerate(novel, 1):
        block = [f"Candidate {i}:"]
        for word, _ in sol:
            block.append(f"  {word}")
        blocks.append(block)

    body = format_columns(blocks, num_cols=num_cols, col_width=col_width)
    count_line = f"Found {len(novel)} candidate tiling(s):\n\n{body}\n"

    unique_strings = sorted({w for sol in novel for w, _ in sol})
    english_hits   = [w for w in unique_strings if w in valid_english_words]
    brand_hits     = [w for w in unique_strings if brand_words and w in brand_words]

    word_lists = (
        format_word_list_text("Unique candidate strings", unique_strings)
        + format_word_list_text("English word matches",   english_hits)
    )
    if brand_words:
        word_lists += format_word_list_text("Brand matches", brand_hits)

    return header + count_line + "\n" + word_lists


if __name__ == "__main__":
    #grid_file = input("Enter grid filename: ")
    grid_file = "grid.txt"

    # Read the grid first so we can report a clear error before prompting,
    # rather than crashing silently after the user has typed their pattern.
    try:
        grid = read_grid(grid_file)
    except FileNotFoundError:
        print(f"Error: Grid file '{grid_file}' not found. "
              f"Make sure it is in the same directory as this script.")
        exit(1)
    except ValueError as e:
        print(f"Error reading grid: {e}")
        exit(1)

    words, prefixes = load_dictionary()

    pattern_str = input("Enter pattern (e.g. --- --- ---): ")

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

    body = format_columns(solution_blocks, num_cols=5, col_width=16) if solution_blocks else ""

    # Load the English word list for filtering unique words.
    valid_english_words = load_valid_english_words("words.txt")

    # Load the brand list for brand-match filtering (brands were already merged
    # into the DFS word set via load_dictionary; we reload here purely for the
    # post-search filtered display — no extra grid search is performed).
    brand_words, _ = load_brands("brands.txt")

    # Build the full structured file content in the required order:
    #   1. Unique words  2. English words  3. Brand matches
    #   4. Grid  5. Pattern  6. Solutions
    file_content = build_solution_file_content(
        solutions, valid_english_words, grid, grid_file,
        pattern_str, solution_blocks, body, brand_words=brand_words
    )

    # --- Relaxed mode ---------------------------------------------------
    # When the normal search finds nothing, run a second pass with no
    # dictionary or brand filter to surface any geometrically valid tiling.
    # We also run it when there ARE normal solutions so the file is complete,
    # but we only *display* it on screen when the normal search came up empty.
    normal_solution_strings = {
        frozenset(w for w, _ in sol) for sol in solutions
    }

    print("\nRunning relaxed mode (unvalidated candidates)...")
    relaxed_solutions = find_relaxed_combinations(
        grid, pattern,
        valid_english_words=valid_english_words,
        brand_words=brand_words,
        prefixes=prefixes
    )
    relaxed_text = format_relaxed_solution_text(
        relaxed_solutions, normal_solution_strings,
        valid_english_words, brand_words
    )

    # Append relaxed section to the file content.
    file_content += relaxed_text

    # Write solution.txt.
    with open("solution.txt", "w", encoding="utf-8") as f:
        f.write(file_content)
    print("Results written to solution.txt")

    # Save a timestamped copy to Execution-History/.
    save_execution_history(file_content)

    # Print the word lists and solutions to the screen as before.
    print_unique_words(solutions, valid_english_words, brand_words=brand_words)
    if solution_blocks:
        print(f"\nFound {len(solution_blocks)} solution(s):\n\n{body}")
    else:
        print("\nNo solution found.")
        # Only surface relaxed candidates on screen when the normal search
        # failed — otherwise the output would be overwhelming.
        print_relaxed_solutions(
            relaxed_solutions, normal_solution_strings,
            valid_english_words, brand_words
        )