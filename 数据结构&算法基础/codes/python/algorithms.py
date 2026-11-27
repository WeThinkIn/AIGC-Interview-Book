"""Deterministic interview algorithms. Contracts and costs: ../README.md.

Original implementations; references are documented in ../REFERENCES.md.
Integer-token APIs deliberately avoid language-dependent Unicode offsets.
"""
from __future__ import annotations
from collections import OrderedDict, deque
from dataclasses import dataclass
import heapq
import math
from itertools import islice

INF = (1 << 63) - 1
DIST_LIMIT = (1 << 62) - 1


@dataclass(slots=True, eq=False)
class ListNode:
    value: int
    next: ListNode | None = None


def middle_node(head: ListNode | None) -> ListNode | None:
    """Acyclic list; the second middle for even lengths; no mutation."""
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
    return slow


def cycle_entry(head: ListNode | None) -> ListNode | None:
    slow = fast = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None


class LRUCache:
    """Single-threaded int -> int cache; miss is None, zero capacity is valid."""
    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError('negative capacity')
        self.capacity = capacity
        self._items: OrderedDict[int, int] = OrderedDict()

    def get(self, key: int) -> int | None:
        try:
            value = self._items[key]
        except KeyError:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return
        self._items[key] = value
        self._items.move_to_end(key)
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)


def next_greater_index(a: list[int]) -> list[int]:
    answer, stack = [-1] * len(a), []
    for i, x in enumerate(a):
        while stack and a[stack[-1]] < x:
            answer[stack.pop()] = i
        stack.append(i)
    return answer


def window_max(a: list[int], k: int) -> list[int]:
    if not 1 <= k <= len(a):
        raise ValueError('require 1 <= k <= n')
    q, answer = deque(), []
    for i, x in enumerate(a):
        while q and q[0] <= i - k:
            q.popleft()
        while q and a[q[-1]] <= x:
            q.pop()
        q.append(i)
        if i >= k - 1:
            answer.append(a[q[0]])
    return answer


def merge_sort(values: list[int]) -> list[int]:
    """Stable, non-mutating, bottom-up merge; two n-element buffers."""
    a, n = list(values), len(values)
    temp = [0] * n
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid, hi = min(lo + width, n), min(lo + 2 * width, n)
            i, j = lo, mid
            for out in range(lo, hi):
                if i < mid and (j >= hi or a[i] <= a[j]):
                    temp[out] = a[i]
                    i += 1
                else:
                    temp[out] = a[j]
                    j += 1
        a, temp = temp, a
        width *= 2
    return a


def lower_bound(a: list[int], target: int) -> int:
    lo, hi = 0, len(a)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def top_k_heap(a: list[int], k: int) -> list[int]:
    """Largest min(k,n) values in descending order, duplicates retained."""
    if k < 0:
        raise ValueError('negative k')
    k = min(k, len(a))
    if k == 0:
        return []
    if k == len(a):
        return sorted(a, reverse=True)
    it = iter(a)
    h = list(islice(it, k))
    heapq.heapify(h)
    replace, threshold = heapq.heapreplace, h[0]
    for x in it:
        if x > threshold:
            replace(h, x)
            threshold = h[0]
    h.sort(reverse=True)
    return h


def top_k(a: list[int], k: int) -> list[int]:
    """Measured policy: native sort for k >= n/4, bounded heap otherwise.

    The threshold is a documented heuristic, not a universal optimum.
    """
    if k < 0:
        raise ValueError('negative k')
    if k == 0:
        return []
    if k * 4 >= len(a):
        return sorted(a, reverse=True)[:k]
    return top_k_heap(a, k)


@dataclass(slots=True, eq=False)
class TreeNode:
    value: int
    left: TreeNode | None = None
    right: TreeNode | None = None


def tree_orders(root: TreeNode | None) -> tuple[list[int], list[int], list[int]]:
    """Preorder, inorder, postorder; explicit frames avoid recursion limits."""
    orders = ([], [], [])
    stack = [(root, 0)] if root is not None else []
    while stack:
        node, phase = stack.pop()
        orders[phase].append(node.value)
        if phase < 2:
            stack.append((node, phase + 1))
            child = node.left if phase == 0 else node.right
            if child is not None:
                stack.append((child, 0))
    return orders


def level_order(root: TreeNode | None) -> list[list[int]]:
    q, result = deque([root]) if root is not None else deque(), []
    while q:
        row = []
        for _ in range(len(q)):
            node = q.popleft()
            row.append(node.value)
            if node.left is not None:
                q.append(node.left)
            if node.right is not None:
                q.append(node.right)
        result.append(row)
    return result


def _graph(n: int, edges: list[tuple[int, int]], undirected=False):
    if n < 0:
        raise ValueError('negative vertex count')
    g = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError('vertex out of range')
        g[u].append(v)
        if undirected:
            g[v].append(u)
    return g


def is_bipartite(n: int, edges: list[tuple[int, int]]) -> bool:
    g, colors = _graph(n, edges, True), [-1] * n
    for start in range(n):
        if colors[start] >= 0:
            continue
        colors[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in g[u]:
                if colors[v] < 0:
                    colors[v] = colors[u] ^ 1
                    q.append(v)
                elif colors[v] == colors[u]:
                    return False
    return True


def topological_sort(n: int, edges: list[tuple[int, int]]) -> list[int] | None:
    g = _graph(n, edges)
    degree = [0] * n
    for _, v in edges:
        degree[v] += 1
    q = deque(i for i in range(n) if degree[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in g[u]:
            degree[v] -= 1
            if degree[v] == 0:
                q.append(v)
    return order if len(order) == n else None


def dijkstra(n: int, edges: list[tuple[int, int, int]], source: int) -> list[int]:
    if n <= 0 or not 0 <= source < n:
        raise ValueError('invalid source')
    g = [[] for _ in range(n)]
    for u, v, w in edges:
        if not (0 <= u < n and 0 <= v < n and 0 <= w <= DIST_LIMIT // n):
            raise ValueError('invalid edge or weight beyond safe domain')
        g[u].append((v, w))
    dist, q = [INF] * n, [(0, source)]
    dist[source] = 0
    while q:
        d, u = heapq.heappop(q)
        if d != dist[u]:
            continue
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(q, (nd, v))
    return dist


def prefix_function(pattern: list[int]) -> list[int]:
    pi = [0] * len(pattern)
    for i in range(1, len(pattern)):
        j = pi[i - 1]
        while j and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
        pi[i] = j
    return pi


def kmp_search(text: list[int], pattern: list[int]) -> int:
    if not pattern:
        return 0
    pi, j = prefix_function(pattern), 0
    for i, token in enumerate(text):
        while j and token != pattern[j]:
            j = pi[j - 1]
        if token == pattern[j]:
            j += 1
        if j == len(pattern):
            return i - len(pattern) + 1
    return -1


class Trie:
    """Integer-token trie; iterative arena, no eager allocation on hits."""
    def __init__(self):
        self.children: list[dict[int, int]] = [{}]
        self.ends = [False]

    def insert(self, word: list[int]) -> None:
        node = 0
        for token in word:
            child = self.children[node].get(token)
            if child is None:
                child = len(self.children)
                self.children[node][token] = child
                self.children.append({})
                self.ends.append(False)
            node = child
        self.ends[node] = True

    def _find(self, word):
        node = 0
        for token in word:
            node = self.children[node].get(token)
            if node is None:
                return None
        return node

    def contains(self, word: list[int]) -> bool:
        node = self._find(word)
        return node is not None and self.ends[node]

    def starts_with(self, prefix: list[int]) -> bool:
        return self._find(prefix) is not None


def _polygon(points):
    if len(points) > 100_000 or any(abs(x) > 1_000_000 or abs(y) > 1_000_000 for x, y in points):
        raise ValueError('geometry domain exceeded')


def polygon_area2(points: list[tuple[int, int]]) -> int:
    """Exact twice-area for a simple integer polygon, no holes."""
    _polygon(points)
    if len(points) < 3:
        return 0
    return abs(sum(x * points[(i + 1) % len(points)][1] - y * points[(i + 1) % len(points)][0]
                   for i, (x, y) in enumerate(points)))


def point_in_polygon(point: tuple[int, int], points: list[tuple[int, int]]) -> int:
    """-1 outside, 0 boundary, +1 inside; exact half-open ray crossings."""
    _polygon(points)
    _polygon([point])
    if len(points) < 3:
        raise ValueError('polygon needs at least three vertices')
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i + 1) % len(points)]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross == 0 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return 0
        if (ay > y) != (by > y) and ((cross > 0) == (by > ay)):
            inside = not inside
    return 1 if inside else -1


def combinations(n: int, k: int) -> list[list[int]]:
    """Lexicographic k-subsets of 1..n; iterative successor avoids stack use."""
    if n < 0 or k < 0:
        raise ValueError('negative n or k')
    if k > n:
        return []
    if k == 0:
        return [[]]
    row, result = list(range(1, k + 1)), []
    while True:
        result.append(row.copy())
        i = k - 1
        while i >= 0 and row[i] == n - k + i + 1:
            i -= 1
        if i < 0:
            return result
        row[i] += 1
        for j in range(i + 1, k):
            row[j] = row[j - 1] + 1


def combinations_backtrack(n: int, k: int) -> list[list[int]]:
    """Pruned recursive teaching variant; k <= 128, use combinations for deep paths."""
    if n < 0 or k < 0:
        raise ValueError('negative n or k')
    if k > n:
        return []
    if k > 128:
        raise ValueError('use iterative combinations for k > 128')
    path, result = [], []
    def visit(start):
        need = k - len(path)
        if need == 0:
            result.append(path.copy())
            return
        for value in range(start, n - need + 2):
            path.append(value)
            visit(value + 1)
            path.pop()
    visit(1)
    return result


def edit_distance(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        diagonal, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            above = row[j]
            row[j] = min(above + 1, row[j - 1] + 1, diagonal + (x != y))
            diagonal = above
    return row[-1]


def sqrt_newton(x: float) -> float:
    if not math.isfinite(x) or x < 0:
        raise ValueError('require finite non-negative x')
    if x == 0:
        return x
    m, e = math.frexp(x)
    if e % 2:
        m, e = m * 2, e - 1
    t = 1.0
    for _ in range(16):
        nxt = .5 * (t + m / t)
        if abs(nxt - t) <= 2 * math.ulp(1.0) * nxt:
            return math.ldexp(nxt, e // 2)
        t = nxt
    raise ArithmeticError('Newton iteration did not converge')


def gaussian_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Row-equilibrated partial pivoting. Rejects numerically small pivots."""
    n = len(a)
    if len(b) != n or any(len(row) != n for row in a):
        raise ValueError('shape mismatch')
    aug = []
    for row, rhs in zip(a, b):
        if not all(math.isfinite(x) for x in [*row, rhs]):
            raise ValueError('nonfinite input')
        scale = max(map(abs, row), default=0.)
        if scale == 0:
            raise ValueError('numerically singular')
        aug.append([x / scale for x in row] + [rhs / scale])
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= 1e-12:
            raise ValueError('numerically singular')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, n):
            f = aug[r][col] / aug[col][col]
            aug[r][col] = 0.
            for c in range(col + 1, n + 1):
                aug[r][c] -= f * aug[col][c]
    x = [0.] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i]
    if not all(map(math.isfinite, x)):
        raise ArithmeticError('nonfinite result')
    return x


class DSU:
    def __init__(self, n: int):
        if n < 0:
            raise ValueError('negative n')
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        if not 0 <= x < len(self.parent):
            raise ValueError('index out of range')
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


class Fenwick:
    """0-based updates; sum of [0,end). Shared cross-language domain: <=2^62 absolute mass."""
    def __init__(self, n: int):
        if n < 0:
            raise ValueError('negative n')
        self.tree = [0] * (n + 1)

    def add(self, index: int, delta: int) -> None:
        if not 0 <= index < len(self.tree) - 1:
            raise ValueError('index out of range')
        i = index + 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, end: int) -> int:
        if not 0 <= end < len(self.tree):
            raise ValueError('end out of range')
        total = 0
        while end:
            total += self.tree[end]
            end -= end & -end
        return total
