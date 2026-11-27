#include "../cpp/algorithms.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <cassert>
using namespace interview;
constexpr int N = 32768;
Values input_values() {
    std::uint32_t s = 20261127;
    Values out;
    out.reserve(N);
    for (int i = 0; i < N; ++i) {
        s = 1664525U * s + 1013904223U;
        out.push_back(s % 100000);
    }
    return out;
}
volatile I sink = 0;
template <class F>
void measure(const std::string &task, const std::string &method, F fn, const Values &expected) {
    if (fn() != expected)
        throw std::runtime_error("benchmark correctness");
    for (int i = 0; i < 8; ++i) {
        auto a = fn();
        sink = a.empty() ? 0 : a[0];
    }
    std::cout << "{\"task\":\"" << task << "\",\"method\":\"" << method << "\",\"samples_ms\":[";
    for (int i = 0; i < 11; ++i) {
        auto t = std::chrono::steady_clock::now();
        auto a = fn();
        auto end = std::chrono::steady_clock::now();
        sink = a.empty() ? 0 : a[0];
        if (i)
            std::cout << ',';
        std::cout << std::chrono::duration<double, std::milli>(end - t).count();
    }
    std::cout << "],\"sink\":" << sink << "}\n";
}
Values sorted(Values a) {
    std::stable_sort(a.begin(), a.end());
    return a;
}
Values full_top(const Values &a, int k) {
    auto b = a;
    std::sort(b.begin(), b.end(), std::greater<I>());
    b.resize(k);
    return b;
}
Values select_top(const Values &a, int k) {
    auto b = a;
    if (k < static_cast<int>(b.size()))
        std::nth_element(b.begin(), b.begin() + k, b.end(), std::greater<I>());
    b.resize(k);
    std::sort(b.begin(), b.end(), std::greater<I>());
    return b;
}
Values pq_top(const Values &a, int k) {
    std::priority_queue<I, Values, std::greater<I>> q;
    for (auto x : a) {
        if (static_cast<int>(q.size()) < k)
            q.push(x);
        else if (x > q.top()) {
            q.pop();
            q.push(x);
        }
    }
    Values out;
    while (!q.empty()) {
        out.push_back(q.top());
        q.pop();
    }
    std::reverse(out.begin(), out.end());
    return out;
}
Values scan(const Values &a, int k) {
    Values out;
    out.reserve(a.size() - k + 1);
    for (std::size_t i = 0; i + k <= a.size(); ++i)
        out.push_back(*std::max_element(a.begin() + i, a.begin() + i + k));
    return out;
}
int main() {
    auto raw = input_values();
    for (std::string kind : {"random", "sorted", "reversed", "duplicates"}) {
        auto a = raw;
        if (kind == "sorted" || kind == "reversed")
            a = sorted(a);
        if (kind == "reversed")
            std::reverse(a.begin(), a.end());
        if (kind == "duplicates")
            for (auto &x : a)
                x %= 8;
        auto expected = sorted(a);
        measure("sort_" + kind, "merge_buffers", [&] { return merge_sort(a); }, expected);
        measure("sort_" + kind, "std_stable_sort", [&] { return sorted(a); }, expected);
    }
    for (int k : {8, 128, N / 2}) {
        auto expected = full_top(raw, k);
        auto task = "top_" + std::to_string(k);
        measure(task, "bounded_heap", [&] { return top_k_heap(raw, k); }, expected);
        measure(task, "adaptive", [&] { return top_k(raw, k); }, expected);
        measure(task, "priority_queue", [&] { return pq_top(raw, k); }, expected);
        measure(task, "nth_then_sort", [&] { return select_top(raw, k); }, expected);
        measure(task, "full_sort", [&] { return full_top(raw, k); }, expected);
    }
    auto expected = scan(raw, 128);
    measure("window_128", "monotonic_ring", [&] { return window_max(raw, 128); }, expected);
    measure("window_128", "scan_each_window", [&] { return scan(raw, 128); }, expected);
}
