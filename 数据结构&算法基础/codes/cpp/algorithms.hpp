#pragma once
// Original C++20 implementations; see ../README.md for contracts.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace interview {
using I = std::int64_t;
using Values = std::vector<I>;
using Edge = std::pair<int, int>;
using WeightedEdge = std::tuple<int, int, I>;
constexpr I INF = std::numeric_limits<I>::max();
constexpr I DIST_LIMIT = (I{1} << 62) - 1;
inline void require(bool ok, const char *why) {
    if (!ok)
        throw std::invalid_argument(why);
}

struct ListNode {
    I value;
    ListNode *next = nullptr;
};
inline ListNode *middle_node(ListNode *head) {
    auto *slow = head;
    auto *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
inline ListNode *cycle_entry(ListNode *head) {
    auto *slow = head;
    auto *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;
        }
    }
    return nullptr;
}

class LRUCache {
    std::size_t capacity_;
    std::list<std::pair<I, I>> items_;
    std::unordered_map<I, decltype(items_)::iterator> positions_;

  public:
    explicit LRUCache(int capacity) : capacity_(capacity < 0 ? 0 : capacity) {
        require(capacity >= 0, "negative capacity");
    }
    // Copying map iterators would leave them bound to the source list.
    LRUCache(const LRUCache &) = delete;
    LRUCache &operator=(const LRUCache &) = delete;
    std::optional<I> get(I key) {
        auto it = positions_.find(key);
        if (it == positions_.end())
            return std::nullopt;
        items_.splice(items_.end(), items_, it->second);
        return it->second->second;
    }
    void put(I key, I value) {
        if (!capacity_)
            return;
        auto it = positions_.find(key);
        if (it != positions_.end()) {
            it->second->second = value;
            items_.splice(items_.end(), items_, it->second);
            return;
        }
        items_.emplace_back(key, value);
        try {
            positions_.emplace(key, std::prev(items_.end()));
        } catch (...) {
            items_.pop_back();
            throw;
        }
        if (items_.size() > capacity_) {
            positions_.erase(items_.front().first);
            items_.pop_front();
        }
    }
};

inline Values next_greater_index(const Values &a) {
    Values result(a.size(), -1);
    std::vector<std::size_t> stack;
    stack.reserve(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        while (!stack.empty() && a[stack.back()] < a[i]) {
            result[stack.back()] = static_cast<I>(i);
            stack.pop_back();
        }
        stack.push_back(i);
    }
    return result;
}
inline Values window_max(const Values &a, int k) {
    require(k > 0 && static_cast<std::size_t>(k) <= a.size(), "require 1 <= k <= n");
    // A k-slot ring stores only live candidate indices.
    std::vector<std::size_t> q(k);
    std::size_t head = 0, count = 0, width = k;
    Values result;
    result.reserve(a.size() - width + 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (count && i >= width && q[head] <= i - width) {
            head = (head + 1) % width;
            --count;
        }
        while (count && a[q[(head + count - 1) % width]] <= a[i])
            --count;
        q[(head + count) % width] = i;
        ++count;
        if (i + 1 >= width)
            result.push_back(a[q[head]]);
    }
    return result;
}
inline Values merge_sort(Values a) {
    const auto n = a.size();
    Values tmp(n);
    for (std::size_t width = 1; width < n;) {
        for (std::size_t lo = 0; lo < n; lo += std::min(2 * width, n - lo)) {
            auto mid = lo + std::min(width, n - lo), hi = mid + std::min(width, n - mid);
            auto i = lo, j = mid;
            for (auto out = lo; out < hi; ++out)
                tmp[out] = (i < mid && (j >= hi || a[i] <= a[j])) ? a[i++] : a[j++];
        }
        a.swap(tmp);
        if (width > n / 2)
            break;
        width *= 2;
    }
    return a;
}
inline std::size_t lower_bound(const Values &a, I target) {
    std::size_t lo = 0, hi = a.size();
    while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (a[mid] < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}
inline Values top_k_heap(const Values &a, int k) {
    require(k >= 0, "negative k");
    auto size = std::min(static_cast<std::size_t>(k), a.size());
    Values h(a.begin(), a.begin() + size);
    if (!size)
        return h;
    if (size == a.size()) {
        std::sort(h.begin(), h.end(), std::greater<I>());
        return h;
    }
    std::make_heap(h.begin(), h.end(), std::greater<I>());
    for (auto i = size; i < a.size(); ++i) {
        if (a[i] <= h[0])
            continue;
        // Replace root with one sift-down, avoiding pop + push traversals.
        std::size_t p = 0;
        while (2 * p + 1 < size) {
            auto child = 2 * p + 1;
            if (child + 1 < size && h[child + 1] < h[child])
                ++child;
            if (h[child] >= a[i])
                break;
            h[p] = h[child];
            p = child;
        }
        h[p] = a[i];
    }
    std::sort(h.begin(), h.end(), std::greater<I>());
    return h;
}
inline Values top_k(const Values &a, int k) {
    require(k >= 0, "negative k");
    auto size = std::min(static_cast<std::size_t>(k), a.size());
    if (!size)
        return {};
    // This measured threshold is a policy, not an architecture-independent optimum.
    if (size >= a.size() / 4) {
        auto out = a;
        if (size < out.size())
            std::nth_element(out.begin(), out.begin() + size, out.end(), std::greater<I>());
        out.resize(size);
        std::sort(out.begin(), out.end(), std::greater<I>());
        return out;
    }
    return top_k_heap(a, k);
}

struct TreeNode {
    I value;
    TreeNode *left = nullptr;
    TreeNode *right = nullptr;
};
inline std::array<Values, 3> tree_orders(const TreeNode *root) {
    std::array<Values, 3> out;
    std::vector<std::pair<const TreeNode *, int>> stack;
    if (root)
        stack.emplace_back(root, 0);
    while (!stack.empty()) {
        auto [node, phase] = stack.back();
        stack.pop_back();
        out[phase].push_back(node->value);
        if (phase < 2) {
            stack.emplace_back(node, phase + 1);
            auto *child = phase == 0 ? node->left : node->right;
            if (child)
                stack.emplace_back(child, 0);
        }
    }
    return out;
}
inline std::vector<Values> level_order(const TreeNode *root) {
    std::deque<const TreeNode *> q;
    std::vector<Values> out;
    if (root)
        q.push_back(root);
    while (!q.empty()) {
        auto size = q.size();
        Values row;
        row.reserve(size);
        while (size--) {
            auto *node = q.front();
            q.pop_front();
            row.push_back(node->value);
            if (node->left)
                q.push_back(node->left);
            if (node->right)
                q.push_back(node->right);
        }
        out.push_back(std::move(row));
    }
    return out;
}
inline std::vector<std::vector<int>> graph(int n, const std::vector<Edge> &edges,
                                           bool undirected = false) {
    require(n >= 0, "negative vertex count");
    std::vector<std::vector<int>> g(n);
    for (auto [u, v] : edges) {
        require(u >= 0 && u < n && v >= 0 && v < n, "vertex out of range");
        g[u].push_back(v);
        if (undirected)
            g[v].push_back(u);
    }
    return g;
}
inline bool is_bipartite(int n, const std::vector<Edge> &edges) {
    auto g = graph(n, edges, true);
    std::vector<int> colors(n, -1), q(n);
    for (int s = 0; s < n; ++s) {
        if (colors[s] >= 0)
            continue;
        std::size_t head = 0, tail = 0;
        colors[s] = 0;
        q[tail++] = s;
        while (head < tail) {
            int u = q[head++];
            for (int v : g[u]) {
                if (colors[v] < 0) {
                    colors[v] = colors[u] ^ 1;
                    q[tail++] = v;
                } else if (colors[v] == colors[u])
                    return false;
            }
        }
    }
    return true;
}
inline std::optional<std::vector<int>> topological_sort(int n, const std::vector<Edge> &edges) {
    auto g = graph(n, edges);
    std::vector<std::size_t> degree(n, 0);
    std::vector<int> order;
    order.reserve(n);
    for (auto [u, v] : edges) {
        (void)u;
        ++degree[v];
    }
    for (int i = 0; i < n; ++i)
        if (degree[i] == 0)
            order.push_back(i);
    for (std::size_t head = 0; head < order.size(); ++head)
        for (int v : g[order[head]])
            if (--degree[v] == 0)
                order.push_back(v);
    if (order.size() != static_cast<std::size_t>(n))
        return std::nullopt;
    return order;
}
inline Values dijkstra(int n, const std::vector<WeightedEdge> &edges, int source) {
    require(n > 0 && source >= 0 && source < n, "invalid source");
    std::vector<std::vector<std::pair<int, I>>> g(n);
    for (auto [u, v, w] : edges) {
        require(u >= 0 && u < n && v >= 0 && v < n && w >= 0 && w <= DIST_LIMIT / n,
                "invalid edge or unsafe weight");
        g[u].emplace_back(v, w);
    }
    Values dist(n, INF);
    dist[source] = 0;
    using Item = std::pair<I, int>;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> q;
    q.emplace(0, source);
    while (!q.empty()) {
        auto [d, u] = q.top();
        q.pop();
        if (d != dist[u])
            continue;
        for (auto [v, w] : g[u])
            if (d + w < dist[v]) {
                dist[v] = d + w;
                q.emplace(dist[v], v);
            }
    }
    return dist;
}
inline std::vector<std::size_t> prefix_function(const Values &p) {
    std::vector<std::size_t> pi(p.size());
    for (std::size_t i = 1; i < p.size(); ++i) {
        auto j = pi[i - 1];
        while (j && p[i] != p[j])
            j = pi[j - 1];
        if (p[i] == p[j])
            ++j;
        pi[i] = j;
    }
    return pi;
}
inline I kmp_search(const Values &text, const Values &p) {
    if (p.empty())
        return 0;
    auto pi = prefix_function(p);
    std::size_t j = 0;
    for (std::size_t i = 0; i < text.size(); ++i) {
        while (j && text[i] != p[j])
            j = pi[j - 1];
        if (text[i] == p[j])
            ++j;
        if (j == p.size())
            return static_cast<I>(i + 1 - p.size());
    }
    return -1;
}
class Trie {
    struct Node {
        std::unordered_map<I, std::size_t> children;
        bool end = false;
    };
    std::vector<Node> nodes_{1};
    std::optional<std::size_t> find(const Values &word) const {
        std::size_t p = 0;
        for (auto token : word) {
            auto it = nodes_[p].children.find(token);
            if (it == nodes_[p].children.end())
                return {};
            p = it->second;
        }
        return p;
    }

  public:
    void insert(const Values &word) {
        std::size_t p = 0;
        for (auto token : word) {
            auto it = nodes_[p].children.find(token);
            if (it == nodes_[p].children.end()) {
                auto child = nodes_.size();
                nodes_.emplace_back();
                // Reacquire the parent after arena growth invalidates references.
                nodes_[p].children.emplace(token, child);
                p = child;
            } else
                p = it->second;
        }
        nodes_[p].end = true;
    }
    bool contains(const Values &word) const {
        auto p = find(word);
        return p && nodes_[*p].end;
    }
    bool starts_with(const Values &word) const {
        return find(word).has_value();
    }
};
using Point = std::pair<I, I>;
inline void check_point(Point p) {
    require(p.first >= -1000000 && p.first <= 1000000 && p.second >= -1000000 &&
                p.second <= 1000000,
            "geometry domain exceeded");
}
inline void check_polygon(const std::vector<Point> &p) {
    require(p.size() <= 100000, "too many vertices");
    for (auto x : p)
        check_point(x);
}
inline I polygon_area2(const std::vector<Point> &p) {
    check_polygon(p);
    if (p.size() < 3)
        return 0;
    I sum = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        auto [x, y] = p[i];
        auto [u, v] = p[(i + 1) % p.size()];
        sum += x * v - y * u;
    }
    return sum < 0 ? -sum : sum;
}
inline int point_in_polygon(Point p, const std::vector<Point> &poly) {
    check_polygon(poly);
    check_point(p);
    require(poly.size() >= 3, "polygon needs three vertices");
    auto [x, y] = p;
    bool inside = false;
    for (std::size_t i = 0; i < poly.size(); ++i) {
        auto [ax, ay] = poly[i];
        auto [bx, by] = poly[(i + 1) % poly.size()];
        I cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
        if (cross == 0 && x >= std::min(ax, bx) && x <= std::max(ax, bx) && y >= std::min(ay, by) &&
            y <= std::max(ay, by))
            return 0;
        if ((ay > y) != (by > y) && ((cross > 0) == (by > ay)))
            inside = !inside;
    }
    return inside ? 1 : -1;
}
inline std::vector<std::vector<int>> combinations(int n, int k) {
    require(n >= 0 && k >= 0, "negative n or k");
    if (k > n)
        return {};
    if (k == 0)
        return {{}};
    std::vector<int> row(k);
    std::iota(row.begin(), row.end(), 1);
    std::vector<std::vector<int>> out;
    while (true) {
        out.push_back(row);
        int i = k - 1;
        while (i >= 0 && row[i] == n - k + i + 1)
            --i;
        if (i < 0)
            return out;
        ++row[i];
        for (int j = i + 1; j < k; ++j)
            row[j] = row[j - 1] + 1;
    }
}
inline std::size_t edit_distance(const Values &aa, const Values &bb) {
    const auto &a = aa.size() >= bb.size() ? aa : bb;
    const auto &b = aa.size() >= bb.size() ? bb : aa;
    std::vector<std::size_t> row(b.size() + 1);
    std::iota(row.begin(), row.end(), 0);
    for (std::size_t i = 1; i <= a.size(); ++i) {
        auto diagonal = row[0];
        row[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            auto above = row[j];
            row[j] = std::min({above + 1, row[j - 1] + 1, diagonal + (a[i - 1] != b[j - 1])});
            diagonal = above;
        }
    }
    return row.back();
}
inline std::vector<std::vector<int>> combinations_backtrack(int n, int k) {
    require(n >= 0 && k >= 0, "negative n or k");
    if (k > n)
        return {};
    require(k <= 128, "use iterative combinations for k > 128");
    std::vector<int> path;
    path.reserve(k);
    std::vector<std::vector<int>> out;
    auto visit = [&](auto &&self, I start) -> void {
        int need = k - static_cast<int>(path.size());
        if (!need) {
            out.push_back(path);
            return;
        }
        for (I value = start; value <= static_cast<I>(n) - need + 1; ++value) {
            path.push_back(static_cast<int>(value));
            self(self, value + 1);
            path.pop_back();
        }
    };
    visit(visit, 1);
    return out;
}
inline double sqrt_newton(double x) {
    require(std::isfinite(x) && x >= 0, "require finite non-negative x");
    if (x == 0)
        return x;
    int e;
    double m = std::frexp(x, &e);
    if (e % 2 != 0) {
        m *= 2;
        --e;
    }
    double t = 1;
    for (int i = 0; i < 16; ++i) {
        double next = .5 * (t + m / t);
        if (std::abs(next - t) <= 2 * std::numeric_limits<double>::epsilon() * next)
            return std::ldexp(next, e / 2);
        t = next;
    }
    throw std::runtime_error("Newton iteration did not converge");
}
inline std::vector<double> gaussian_solve(const std::vector<std::vector<double>> &a,
                                          const std::vector<double> &b) {
    const auto n = a.size();
    require(b.size() == n, "shape mismatch");
    auto aug = a;
    for (std::size_t r = 0; r < n; ++r) {
        require(a[r].size() == n && std::isfinite(b[r]), "shape or nonfinite input");
        double scale = 0;
        for (auto x : a[r]) {
            require(std::isfinite(x), "nonfinite input");
            scale = std::max(scale, std::abs(x));
        }
        require(scale > 0, "numerically singular");
        for (auto &x : aug[r])
            x /= scale;
        aug[r].push_back(b[r] / scale);
    }
    for (std::size_t col = 0; col < n; ++col) {
        auto pivot = col;
        for (auto r = col + 1; r < n; ++r)
            if (std::abs(aug[r][col]) > std::abs(aug[pivot][col]))
                pivot = r;
        require(std::abs(aug[pivot][col]) > 1e-12, "numerically singular");
        std::swap(aug[col], aug[pivot]);
        for (auto r = col + 1; r < n; ++r) {
            double f = aug[r][col] / aug[col][col];
            aug[r][col] = 0;
            for (auto c = col + 1; c <= n; ++c)
                aug[r][c] -= f * aug[col][c];
        }
    }
    std::vector<double> x(n);
    for (auto i = n; i-- > 0;) {
        double sum = 0;
        for (auto j = i + 1; j < n; ++j)
            sum += aug[i][j] * x[j];
        x[i] = (aug[i][n] - sum) / aug[i][i];
        require(std::isfinite(x[i]), "nonfinite result");
    }
    return x;
}
class DSU {
    std::vector<int> parent_, size_;

  public:
    explicit DSU(int n) {
        require(n >= 0, "negative n");
        parent_.resize(n);
        size_.assign(n, 1);
        std::iota(parent_.begin(), parent_.end(), 0);
    }
    int find(int x) {
        require(x >= 0 && static_cast<std::size_t>(x) < parent_.size(), "index out of range");
        while (x != parent_[x]) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b)
            return false;
        if (size_[a] < size_[b])
            std::swap(a, b);
        parent_[b] = a;
        size_[a] += size_[b];
        return true;
    }
};
class Fenwick {
    Values tree_;

  public:
    explicit Fenwick(int n) {
        require(n >= 0, "negative n");
        tree_.resize(static_cast<std::size_t>(n) + 1);
    }
    // Caller maintains the documented absolute-mass bound; no unchecked external index.
    void add(int index, I delta) {
        require(index >= 0 && static_cast<std::size_t>(index) + 1 < tree_.size(),
                "index out of range");
        for (std::size_t i = static_cast<std::size_t>(index) + 1; i < tree_.size();
             i += i & (~i + 1))
            tree_[i] += delta;
    }
    I prefix_sum(int end) const {
        require(end >= 0 && static_cast<std::size_t>(end) < tree_.size(), "end out of range");
        I total = 0;
        for (auto i = static_cast<std::size_t>(end); i; i -= i & (~i + 1))
            total += tree_[i];
        return total;
    }
};
} // namespace interview
