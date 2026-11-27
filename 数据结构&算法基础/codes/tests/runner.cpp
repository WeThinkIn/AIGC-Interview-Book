#include "../cpp/algorithms.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstdlib>
using namespace interview;
template <class T> void json(const T &x) {
    std::cout << x;
}
void json(bool x) {
    std::cout << (x ? "true" : "false");
}
template <class T> void json(const std::vector<T> &a) {
    std::cout << '[';
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i)
            std::cout << ',';
        json(a[i]);
    }
    std::cout << ']';
}
void json(const std::vector<bool> &a) {
    std::cout << '[';
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (i)
            std::cout << ',';
        json(static_cast<bool>(a[i]));
    }
    std::cout << ']';
}
I read(std::istream &s) {
    I x;
    if (!(s >> x))
        throw std::invalid_argument("bad input");
    return x;
}
Values vals(std::istream &s, int n) {
    require(n >= 0, "negative n");
    Values a;
    for (int i = 0; i < n; ++i)
        a.push_back(read(s));
    return a;
}
void execute(std::istringstream &s) {
    std::string op;
    s >> op;
    if (op == "sort" || op == "next" || op == "lower" || op == "top" || op == "top_heap" ||
        op == "window") {
        int n = read(s);
        I k = (op == "lower" || op == "top" || op == "top_heap" || op == "window") ? read(s) : 0;
        auto a = vals(s, n);
        if (op == "sort")
            json(merge_sort(a));
        else if (op == "next")
            json(next_greater_index(a));
        else if (op == "lower")
            json(interview::lower_bound(a, k));
        else if (op == "top")
            json(top_k(a, static_cast<int>(k)));
        else if (op == "top_heap")
            json(top_k_heap(a, static_cast<int>(k)));
        else
            json(window_max(a, static_cast<int>(k)));
        return;
    }
    if (op == "list") {
        int n = read(s), cycle = read(s);
        auto a = vals(s, n);
        std::vector<ListNode> nodes(n);
        for (int i = 0; i < n; ++i) {
            nodes[i].value = a[i];
            if (i + 1 < n)
                nodes[i].next = &nodes[i + 1];
        }
        auto *mid = middle_node(n ? &nodes[0] : nullptr);
        I m = mid ? mid - &nodes[0] : -1;
        if (cycle >= 0)
            nodes.back().next = &nodes[cycle];
        auto *entry = cycle_entry(n ? &nodes[0] : nullptr);
        json(Values{m, entry ? entry - &nodes[0] : -1});
        return;
    }
    if (op == "tree") {
        int n = read(s);
        std::vector<TreeNode> nodes(n);
        for (int i = 0; i < n; ++i) {
            nodes[i].value = read(s);
            int l = read(s), r = read(s);
            nodes[i].left = l < 0 ? nullptr : &nodes[l];
            nodes[i].right = r < 0 ? nullptr : &nodes[r];
        }
        auto *root = n ? &nodes[0] : nullptr;
        auto o = tree_orders(root);
        std::cout << '[';
        for (auto &a : o) {
            json(a);
            std::cout << ',';
        }
        json(level_order(root));
        std::cout << ']';
        return;
    }
    if (op == "lru") {
        LRUCache c(static_cast<int>(read(s)));
        int n = read(s);
        bool first = true;
        std::cout << '[';
        for (int i = 0; i < n; ++i) {
            int kind = read(s);
            I key = read(s), value = read(s);
            if (!kind)
                c.put(key, value);
            else {
                if (!first)
                    std::cout << ',';
                first = false;
                auto v = c.get(key);
                if (v)
                    json(*v);
                else
                    std::cout << "null";
            }
        }
        std::cout << ']';
        return;
    }
    if (op == "bip" || op == "topo") {
        int n = read(s), m = read(s);
        std::vector<Edge> e;
        for (int i = 0; i < m; ++i) {
            int u = read(s), v = read(s);
            e.emplace_back(u, v);
        }
        if (op == "bip")
            json(is_bipartite(n, e));
        else {
            auto x = topological_sort(n, e);
            if (x)
                json(*x);
            else
                std::cout << "null";
        }
        return;
    }
    if (op == "dijkstra") {
        int n = read(s), m = read(s), src = read(s);
        std::vector<WeightedEdge> e;
        for (int i = 0; i < m; ++i) {
            int u = read(s), v = read(s);
            I w = read(s);
            e.emplace_back(u, v, w);
        }
        json(dijkstra(n, e, src));
        return;
    }
    if (op == "kmp" || op == "edit") {
        int n = read(s), m = read(s);
        auto a = vals(s, n), b = vals(s, m);
        if (op == "kmp")
            json(kmp_search(a, b));
        else
            json(edit_distance(a, b));
        return;
    }
    if (op == "trie") {
        int n = read(s);
        Trie t;
        std::vector<bool> out;
        for (int i = 0; i < n; ++i) {
            int kind = read(s), size = read(s);
            auto w = vals(s, size);
            if (!kind)
                t.insert(w);
            else
                out.push_back(kind == 1 ? t.contains(w) : t.starts_with(w));
        }
        json(out);
        return;
    }
    if (op == "polygon") {
        int n = read(s);
        I x = read(s), y = read(s);
        std::vector<Point> p;
        for (int i = 0; i < n; ++i) {
            I a = read(s), b = read(s);
            p.emplace_back(a, b);
        }
        // Compute both before printing so invalid inputs produce one valid error record.
        auto area = polygon_area2(p);
        auto inside = point_in_polygon({x, y}, p);
        json(Values{area, inside});
        return;
    }
    if (op == "comb") {
        int n = read(s), k = read(s);
        json(combinations(n, k));
        return;
    }
    if (op == "comb_backtrack") {
        int n = read(s), k = read(s);
        json(combinations_backtrack(n, k));
        return;
    }
    if (op == "sqrt") {
        std::string v;
        s >> v;
        char *end = nullptr;
        double x = std::strtod(v.c_str(), &end);
        require(end == v.c_str() + v.size(), "bad float");
        json(sqrt_newton(x));
        return;
    }
    if (op == "gauss") {
        int n = read(s);
        std::vector<std::vector<double>> a(n, std::vector<double>(n));
        std::vector<double> b(n);
        for (auto &row : a)
            for (auto &x : row)
                s >> x;
        for (auto &x : b)
            s >> x;
        json(gaussian_solve(a, b));
        return;
    }
    if (op == "dsu") {
        int n = read(s), q = read(s);
        DSU d(n);
        std::vector<bool> out;
        for (int i = 0; i < q; ++i) {
            int kind = read(s), a = read(s), b = read(s);
            out.push_back(!kind ? d.unite(a, b) : d.find(a) == d.find(b));
        }
        json(out);
        return;
    }
    if (op == "fenwick") {
        int n = read(s), q = read(s);
        Fenwick f(n);
        Values out;
        for (int i = 0; i < q; ++i) {
            int kind = read(s), index = read(s);
            I value = read(s);
            if (!kind)
                f.add(index, value);
            else
                out.push_back(f.prefix_sum(index));
        }
        json(out);
        return;
    }
    throw std::invalid_argument("unknown operation");
}
int main() {
    std::cout << std::setprecision(17);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream s(line);
        try {
            execute(s);
        } catch (const std::exception &) {
            std::cout << "{\"error\":true}";
        }
        std::cout << '\n';
    }
}
