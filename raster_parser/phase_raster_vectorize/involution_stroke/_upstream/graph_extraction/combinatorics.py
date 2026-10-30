# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/combinatorics.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

import itertools

import logging as _logging
_LOG = _logging.getLogger("raster_parser.involution_stroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))



def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def _solve_impl(X, Y):
    """
    X: A dictionary of columns {index: set_of_row_keys_covering_it}
    Y: A dictionary of rows {row_key: list_of_indices_it_covers}
    """
    if not X:
        yield []
    else:
        # Choose the column with the fewest rows (heuristic to speed up)
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution = [r]
            cols = select(X, Y, r)
            for res in _solve_impl(X, Y):
                yield solution + res
            deselect(X, Y, r, cols)


def solve_exact_cover(X, Y):
    X = {j: set() for j in X}
    for i in Y:
        assert isinstance(Y[i], list)
        for j in Y[i]:
            X[j].add(i)
    yield from _solve_impl(X, Y)
    
    
import itertools

def subset_cover(X, mp, n):
    """
    Generates combinations of n IDs from mp such that the union of their 
    corresponding lists covers the set X.
    
    Args:
        X (set or list): The universe of elements to cover.
        mp (dict): A map where key -> list of elements (the subsets).
        n (int): The number of subsets to include in each combination.
        
    Yields:
        tuple: A tuple of keys from mp that satisfy the cover condition.
    """
    # 1. Convert target X to a set for O(1) lookups
    target = set(X)
    target_len = len(target)
    
    # 2. Pre-process the map: 
    #    Convert values to sets and filter out elements NOT in X.
    #    This optimizes memory and speeds up the union operation.
    #    We map ID -> (Original Set for reference, Filtered Set for logic)
    pool = {}
    for k, v in mp.items():
        s = set(v)
        pool[k] = s

    # 3. Get all available IDs
    keys = list(pool.keys())
    
    # 4. Iterate over all combinations of size n
    for combo in itertools.combinations(keys, n):
        # Calculate the union of the filtered sets in this combination
        # set().union(...) is efficient in Python
        current_union = set().union(*(pool[k] for k in combo))
        
        # 5. Check coverage
        # Since we filtered 'pool' to only contain elements in 'target',
        # we just need to check if the lengths are equal.
        if len(current_union) == target_len:
            yield combo

