"""
Optimized RPD indices.

Public functions preserve the semantics of the original computeindex.py
for sequences with at least two elements:

    RPDlep, RPDlea, RPDlepr,
    RPDgep, RPDgea, RPDgepr

Additional function:
    progressive_indices(elements)

The progressive function computes the complete trajectories for prefixes
x[:2], x[:3], ..., x[:n] without repeatedly invoking every index from
scratch.

Complexities
------------
Single final value:
    RPDlep/RPDlea/RPDlepr : O(n)
    RPDgep/RPDgea/RPDgepr : O(n log n)

Complete progressive trajectories:
    RPDlep/RPDlea          : O(n)
    RPDgep/RPDgea          : O(n log n)
    RPDlepr/RPDgepr        : O(n^2)

No third-party packages are required.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def isAworstThanB(A, B):
    """Compatibility helper from the original implementation."""
    if A < B:
        return True
    if A > B:
        return False
    return None


def _rpd(P: float, N: float) -> float:
    den = P + N
    return (P - N) / den if den > 0 else 0.0


class _Fenwick:
    """Fenwick tree (Binary Indexed Tree) for prefix sums."""

    __slots__ = ("n", "tree")

    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)

    def add(self, i: int, value):
        tree = self.tree
        n = self.n
        while i <= n:
            tree[i] += value
            i += i & -i

    def prefix_sum(self, i: int):
        tree = self.tree
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & -i
        return total


def _rank_data(elements: Sequence[float]):
    values = sorted(set(elements))
    ranks = {value: i + 1 for i, value in enumerate(values)}
    return ranks, len(values)


# ---------------------------------------------------------------------------
# Local indices
# ---------------------------------------------------------------------------

def RPDlep(elements: Sequence[float]) -> float:
    n = len(elements)
    if n < 2:
        return 0.0

    cp = cn = 0
    ps = ns = 0.0

    previous = elements[0]
    for current in elements[1:]:
        if previous < current:
            cp += 1
            ps += current - previous
        elif previous > current:
            cn += 1
            ns += previous - current
        previous = current

    # In the original formula both weighted terms are divided by
    # comparisonTot=(n-1); that common factor cancels in the final ratio.
    return _rpd(cp * ps, cn * ns)


def RPDlea(elements: Sequence[float]) -> float:
    if len(elements) < 2:
        return 0.0

    cp = cn = 0
    P = N = 0.0
    previous = elements[0]

    for comparison_tot, current in enumerate(elements[1:], start=1):
        if previous < current:
            cp += 1
            P += (current - previous) * cp / comparison_tot
        elif previous > current:
            cn += 1
            N += (previous - current) * cn / comparison_tot
        previous = current

    return _rpd(P, N)


def RPDlepr(elements: Sequence[float]) -> float:
    n = len(elements)
    if n < 2:
        return 0.0

    cp = cn = 0
    P = N = 0.0
    comparison_tot = 0

    for idx in range(n - 1, 0, -1):
        comparison_tot += 1
        a = elements[idx - 1]
        b = elements[idx]

        if a < b:
            cp += 1
            P += (b - a) * cp / comparison_tot
        elif a > b:
            cn += 1
            N += (a - b) * cn / comparison_tot

    return _rpd(P, N)


# ---------------------------------------------------------------------------
# Global indices
# ---------------------------------------------------------------------------

def RPDgep(elements: Sequence[float]) -> float:
    """Global pairwise RPD in O(n log n), instead of O(n^2)."""
    n = len(elements)
    if n < 2:
        return 0.0

    ranks, m = _rank_data(elements)
    counts = _Fenwick(m)
    sums = _Fenwick(m)

    seen = 0
    seen_sum = 0.0

    cp = cn = 0
    ps = ns = 0.0

    for x in elements:
        r = ranks[x]

        less_count = counts.prefix_sum(r - 1)
        less_sum = sums.prefix_sum(r - 1)

        le_count = counts.prefix_sum(r)
        le_sum = sums.prefix_sum(r)

        greater_count = seen - le_count
        greater_sum = seen_sum - le_sum

        # Earlier value < current value: positive pair.
        cp += less_count
        ps += x * less_count - less_sum

        # Earlier value > current value: negative pair.
        cn += greater_count
        ns += greater_sum - x * greater_count

        counts.add(r, 1)
        sums.add(r, x)
        seen += 1
        seen_sum += x

    # The common comparisonTot factor from the original implementation
    # cancels between numerator and denominator.
    return _rpd(cp * ps, cn * ns)


def RPDgea(elements: Sequence[float]) -> float:
    """Global forward-averaged RPD in O(n log n)."""
    n = len(elements)
    if n < 2:
        return 0.0

    ranks, m = _rank_data(elements)
    counts = _Fenwick(m)
    sums = _Fenwick(m)

    seen = 0
    seen_sum = 0.0
    P = N = 0.0

    for j, x in enumerate(elements):
        if j:
            r = ranks[x]

            less_count = counts.prefix_sum(r - 1)
            less_sum = sums.prefix_sum(r - 1)

            le_count = counts.prefix_sum(r)
            le_sum = sums.prefix_sum(r)

            greater_count = seen - le_count
            greater_sum = seen_sum - le_sum

            positive_sum = x * less_count - less_sum
            negative_sum = greater_sum - x * greater_count

            P += positive_sum * less_count / j
            N += negative_sum * greater_count / j

        r = ranks[x]
        counts.add(r, 1)
        sums.add(r, x)
        seen += 1
        seen_sum += x

    return _rpd(P, N)


def RPDgepr(elements: Sequence[float]) -> float:
    """Global reverse-averaged RPD in O(n log n)."""
    n = len(elements)
    if n < 2:
        return 0.0

    ranks, m = _rank_data(elements)
    counts = _Fenwick(m)
    sums = _Fenwick(m)

    seen = 0
    seen_sum = 0.0
    P = N = 0.0

    # Scan from right to left: the Fenwick trees contain the "later" values.
    for later_count_total, x in enumerate(reversed(elements)):
        if later_count_total:
            r = ranks[x]

            less_count = counts.prefix_sum(r - 1)
            less_sum = sums.prefix_sum(r - 1)

            le_count = counts.prefix_sum(r)
            le_sum = sums.prefix_sum(r)

            greater_count = seen - le_count
            greater_sum = seen_sum - le_sum

            # Positive in the original code when current < later.
            positive_sum = greater_sum - x * greater_count
            negative_sum = x * less_count - less_sum

            P += positive_sum * greater_count / later_count_total
            N += negative_sum * less_count / later_count_total

        r = ranks[x]
        counts.add(r, 1)
        sums.add(r, x)
        seen += 1
        seen_sum += x

    return _rpd(P, N)


# ---------------------------------------------------------------------------
# Progressive trajectories
# ---------------------------------------------------------------------------

def progressive_indices(elements: Sequence[float]) -> dict[str, list[float]]:
    """
    Compute all six RPD trajectories for prefixes:

        elements[:2], elements[:3], ..., elements[:n]

    This is substantially faster than:
        [RPDxxx(elements[:k]) for k in range(2, n+1)]
    """
    x = list(elements)
    n = len(x)

    names = ("RPDlep", "RPDlea", "RPDlepr", "RPDgep", "RPDgea", "RPDgepr")
    if n < 2:
        return {name: [] for name in names}

    out = {name: [] for name in names}

    # ---- Local forward: RPDlep + RPDlea in O(n) ----
    cp = cn = 0
    ps = ns = 0.0
    P_a = N_a = 0.0

    for k in range(1, n):
        a = x[k - 1]
        b = x[k]
        comparison_tot = k

        if a < b:
            delta = b - a
            cp += 1
            ps += delta
            P_a += delta * cp / comparison_tot
        elif a > b:
            delta = a - b
            cn += 1
            ns += delta
            N_a += delta * cn / comparison_tot

        out["RPDlep"].append(_rpd(cp * ps, cn * ns))
        out["RPDlea"].append(_rpd(P_a, N_a))

    # ---- RPDlepr: reverse weighting changes when every new point arrives,
    #      therefore O(n^2), but without temporary positive/negative lists. ----
    for end in range(1, n):
        cp_r = cn_r = 0
        P_r = N_r = 0.0
        comparison_tot = 0

        for idx in range(end, 0, -1):
            comparison_tot += 1
            a = x[idx - 1]
            b = x[idx]

            if a < b:
                cp_r += 1
                P_r += (b - a) * cp_r / comparison_tot
            elif a > b:
                cn_r += 1
                N_r += (a - b) * cn_r / comparison_tot

        out["RPDlepr"].append(_rpd(P_r, N_r))

    # ---- Global forward: RPDgep + RPDgea in O(n log n) ----
    ranks, m = _rank_data(x)
    counts = _Fenwick(m)
    sums = _Fenwick(m)

    seen = 0
    seen_sum = 0.0

    pair_cp = pair_cn = 0
    pair_ps = pair_ns = 0.0
    P_gea = N_gea = 0.0

    for j, current in enumerate(x):
        r = ranks[current]

        less_count = counts.prefix_sum(r - 1)
        less_sum = sums.prefix_sum(r - 1)

        le_count = counts.prefix_sum(r)
        le_sum = sums.prefix_sum(r)

        greater_count = seen - le_count
        greater_sum = seen_sum - le_sum

        if j:
            positive_sum = current * less_count - less_sum
            negative_sum = greater_sum - current * greater_count

            pair_cp += less_count
            pair_ps += positive_sum
            pair_cn += greater_count
            pair_ns += negative_sum

            out["RPDgep"].append(
                _rpd(pair_cp * pair_ps, pair_cn * pair_ns)
            )

            P_gea += positive_sum * less_count / j
            N_gea += negative_sum * greater_count / j
            out["RPDgea"].append(_rpd(P_gea, N_gea))

        counts.add(r, 1)
        sums.add(r, current)
        seen += 1
        seen_sum += current

    # ---- RPDgepr progressive in O(n^2).
    # For each old head i, maintain its statistics against later points.
    cpos = [0] * n
    cneg = [0] * n
    spos = [0.0] * n
    sneg = [0.0] * n

    P_total = N_total = 0.0

    for j in range(1, n):
        current = x[j]

        for i in range(j):
            old_den = j - i - 1

            if old_den > 0:
                P_total -= spos[i] * cpos[i] / old_den
                N_total -= sneg[i] * cneg[i] / old_den

            head = x[i]
            if head < current:
                cpos[i] += 1
                spos[i] += current - head
            elif head > current:
                cneg[i] += 1
                sneg[i] += head - current

            new_den = j - i
            P_total += spos[i] * cpos[i] / new_den
            N_total += sneg[i] * cneg[i] / new_den

        out["RPDgepr"].append(_rpd(P_total, N_total))

    return out


if __name__ == "__main__":
    # Minimal self-test.
    samples = [
        [1, 2, 4, 6],
        [6, 4, 2, 1],
        [1, 1, 1, 1],
        [1, 3, 2, 5, 4, 4],
        [0.0183179687636068, 0.66165806279797, 1, 6.51487788578436e-10],
    ]

    for sample in samples:
        print(sample)
        for name in ("RPDlep", "RPDlea", "RPDlepr", "RPDgep", "RPDgea", "RPDgepr"):
            print(f"  {name:8s} = {globals()[name](sample): .12f}")
