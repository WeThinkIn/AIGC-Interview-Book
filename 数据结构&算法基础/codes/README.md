# 数据结构与算法：三语言实现

这套代码把算法原理、接口契约、正确性验证和性能测量放在一起。适合配合[第 7 章](../07_三语言代码模板与AI场景实现高频考点.md)阅读：先解释不变量，再预测边界行为，最后运行测试和基准。C/C++ 方向使用 **C++20** 实现，另外提供 Python 与 Java 版本。

## 文件与运行

- [Python 实现](python/algorithms.py)：Python 3.10 及以上；本次使用 CPython 3.11.13。
- [C++ 实现](cpp/algorithms.hpp)：C++20 单头文件，命名空间 `interview`。
- [Java 实现](java/Algorithms.java)：Java 21，类名 `Algorithms`。
- [统一验证入口](tests/verify.py)：生成相同输入，与独立参考答案比较，并编译运行三语言实现。
- [性能基准](BENCHMARKS.md)：计时范围、输入分布、标准库对照与实际结果。
- [参考实现与来源](REFERENCES.md)：阅读过的项目版本、借鉴范围与许可证。

在本目录下执行，产物放入 `build/`，不会覆盖源码：

```bash
python3 tests/verify.py --out build/verification
python3 benchmarks/run.py --out build/benchmark
```

需要可用的 `clang++`、`java` 和 `javac`。验证脚本支持 `--cxx`、`--java`、`--javac`；基准脚本支持 `--java`、`--javac`，默认使用 clang++，当前机器信息采集针对 macOS。JDK 未在 PATH 中时，可传入对应可执行文件的完整路径。Python 验证与基准脚本仅使用标准库。

验证默认额外运行 C++ AddressSanitizer 与 UndefinedBehaviorSanitizer。平台不支持时可显式传 `--skip-sanitizer`，但此时结果只代表普通构建通过。详细用例、各语言输出与验证报告会保存在指定目录。

Python 调用示例：

```python
import sys
sys.path.insert(0, "python")
from algorithms import top_k, window_max, kmp_search, Fenwick

assert top_k([3, 1, 3, 2], 2) == [3, 3]
assert window_max([1, 3, -1, -3, 5], 3) == [3, 3, 5]
assert kmp_search([65, 0x1F600, 66], [0x1F600]) == 1
tree = Fenwick(3)
tree.add(1, 5)
assert tree.prefix_sum(2) == 5
```

C++ 调用示例：

```cpp
#include "cpp/algorithms.hpp"
#include <cassert>
int main() {
    auto answer = interview::top_k({3, 1, 3, 2}, 2);
    assert((answer == interview::Values{3, 3}));
    interview::LRUCache cache(1);
    cache.put(7, -1);
    assert(cache.get(7).value() == -1);
    assert(!cache.get(8).has_value());
}
```

将其保存为 `demo.cpp` 后，可用 `clang++ -std=c++20 -O3 demo.cpp -o build/demo` 编译；先确保 `build/` 已存在。

Java 调用示例：

```java
import java.util.Arrays;
class Demo {
    public static void main(String[] args) {
        long[] values = Algorithms.topK(new long[]{3, 1, 3, 2}, 2);
        if (!Arrays.equals(values, new long[]{3, 3})) {
            throw new AssertionError("Top K");
        }
    }
}
```

将其保存为 `Demo.java`，运行 `javac -d build java/Algorithms.java Demo.java`，再运行 `java -cp build Demo`。

## 共同契约

序列元素使用有符号 64 位整数的共同取值域。Python 的整数可以更大，但超出该域就不属于三语言一致性保证。长度与下标须能由各语言容器表示并能实际分配。Python 类型注解不承担完整的运行时类型检查，Java 的输入数组、列表及节点字段也应满足非空、类型正确等前置条件。

除容器的更新操作外，算法保留调用者输入；输出为独立结果。链表、树函数只读取结构，返回的节点仍属于原结构。C++ 的节点指针是借用关系，调用者负责分配、释放和有效期；函数不会替调用者删除节点。树必须无环且不共享子节点，中点函数要求链表无环。验证这些结构前提会改变时间或空间成本，因此接口不会暗中扫描所有权关系。

所有容器都按单线程语义提供。需要并发访问时，由调用者选择锁或适合并发的成熟容器。资源耗尽、任意用户对象、浮点分数排序、外部状态事务不在这些整数教学接口中。

核心接口如下。Java 通常使用对应的 camelCase 名称；三个实现的完整签名以源码为准。空间列不含输入和最终输出，图算法另列构图成本。

<div align="center">

| 功能 / Python 与 C++ 名称 | Java 名称 | 主要语义与复杂度 |
|---|---|---|
| `middle_node` / `cycle_entry` | `middleNode` / `cycleEntry` | 中点取偶数长度的后一个；环比较节点身份；O(n) 时间、O(1) 辅助空间 |
| `LRUCache.get/put` | 同名 | 容量 0 合法，负容量报错；miss 为 None / nullopt / null；哈希操作为期望常数时间 |
| `next_greater_index` | `nextGreaterIndex` | 右侧第一个严格更大值的下标；不存在为 -1；O(n) 时间与辅助栈 |
| `window_max` | `windowMax` | 要求 1≤k≤n；O(n) 时间、O(k) 辅助空间 |
| `merge_sort` | `mergeSort` | 稳定、自底向上、保留输入；O(n log n) 时间、O(n) 辅助存储 |
| `lower_bound` | `lowerBound` | 输入升序；第一个不小于目标的位置，允许返回 n；O(log n) 时间 |
| `top_k_heap` / `top_k` | `topKHeap` / `topK` | 降序、保留重复；k=0 空、k>n 全部、k<0 错误；两条性能路线见下文 |
| `tree_orders` / `level_order` | `treeOrders` / `levelOrder` | 前中后序及分层结果；O(n) 时间；DFS 栈 O(h)，BFS 队列 O(w) |
| `is_bipartite` | `isBipartite` | 无向边表，函数补双向邻接；空图为真，自环为假；O(V+E) |
| `topological_sort` | `topologicalSort` | 有向边表；有环为 None / nullopt / null，空图为空序列；O(V+E) |
| `dijkstra` | 同名 | 有向非负权图，返回源点到全部点的距离；惰性堆 O((V+E) log(V+E))，堆可能 O(E) |
| `prefix_function` / `kmp_search` | `prefixFunction` / `kmpSearch` | 整数符号序列；首个命中下标，未命中 -1，空模式 0；KMP O(n+m) |
| `Trie` | 同名 | insert、contains、starts_with；Java 为 startsWith；空前缀为真，空词需显式插入才算存在 |
| `polygon_area2` / `point_in_polygon` | `polygonArea2` / `pointInPolygon` | 无孔简单整数多边形；返回两倍面积，位置为 -1/0/1；O(n) |
| `combinations` / `combinations_backtrack` | `combinations` / `combinationsBacktrack` | 1..n 的 k 元组合，字典序；k=0 为一个空组合，k>n 为空；输出成本 O(k·C(n,k))，k=0 另计 O(1) |
| `edit_distance` | `editDistance` | 插入、删除、替换单位成本；整数符号序列；O(nm) 时间、O(min(n,m)) 辅助空间 |
| `sqrt_newton` / `gaussian_solve` | `sqrtNewton` / `gaussianSolve` | 有限数值域与精度限制见下文；高斯消元 O(n³) 时间、O(n²) 副本 |
| `DSU.find/union` | `find/union`；C++ 为 `unite` | 大小合并与路径压缩；union 返回是否真正合并；摊还 O(α(n)) |
| `Fenwick.add/prefix_sum` | `add/prefixSum` | 0 基下标更新，前缀区间 [0,end)；O(log n) 时间、O(n) 存储 |

</div>

Trie 使用节点池和整数索引，路径遍历及释放不依赖递归。单次长度为 L 的词操作，哈希子节点查询良好分布时为期望 O(L)，总存储与不同前缀数有关。图函数从边表建立邻接表，因此每次调用还需 O(V+E) 时间和存储；有大量重复查询时应把构图与查询分离。

## 容易影响正确性的边界

**文本的下标单位。** KMP、Trie、编辑距离接收整数序列。调用者可选择字节、Unicode 码点或 token ID，但不能在三语言中混用。Python 可用 `[ord(c) for c in text]`，Java 可把 `text.codePoints().toArray()` 转为 long 数组；C++ 需要明确的 UTF-8 解码步骤。字形分段、正规化和大小写折叠不包含在匹配算法中。

**距离与整数溢出。** Dijkstra 要求 V>0，端点与源点合法，边权满足 `0 <= w <= ((1<<62)-1)/V`。此保守域让松弛加法与不可达哨兵分离；不可达统一为 `2^63-1`。Fenwick 要求整个使用期间所有更新的绝对值累计不超过 `2^62`，由调用者保证；这比只检查最终总和更严格，也能约束每个内部结点与前缀累计的中间值。

**几何。** 每个坐标绝对值不超过 1,000,000，顶点数不超过 100,000；函数检查该数值域，但不执行全局自交检查。调用者保证简单、无孔的多边形。`polygon_area2` 对少于 3 点返回 0，点位置函数则拒绝少于 3 点。位置结果 0 明确表示在边或顶点上，射线使用半开跨越规则避免顶点重复计数。浮点坐标、洞和自交图形应使用更完整的几何库。

**数值方法。** 牛顿平方根拒绝负数、NaN 和无穷，保留正负零，先缩放指数后迭代，覆盖次正规数；测试容许与标准平方根最多 3 ULP 的差距，不承诺正确舍入。高斯消元先行均衡、再部分选主元，在缩放后主元绝对值不超过 `1e-12` 时拒绝求解；这是一项数值阈值策略，可能拒绝数学上非奇异但病态的输入。它只提供方阵唯一解路径，既不估计条件数，也不处理一般秩亏/最小二乘问题。空 0×0 系统返回空解，非有限输入或结果报错。生产数值计算优先采用经过验证的 BLAS/LAPACK 等库。

**输出规模。** 组合函数完整返回所有组合，结果数量本身可能很大。递归教学版本限制 k≤128；更深路径可用迭代版本，但迭代无法消除组合爆炸。调用者应先估算输出数量，必要时改成迭代器、回调或带预算的枚举。

## 已做的性能取舍

归并复用两块缓冲，避免头部删除与递归切片；树遍历用阶段栈，避免深递归和反复拼接；C++、Java 的窗口队列使用 k 槽环形数组；Java Top K 使用基本类型数组，减少装箱。C++ LRU 禁止复制，避免哈希表里的迭代器指向另一份链表；新节点加入映射失败时会撤销该节点。

`top_k_heap` 适合 K 小或需要有界候选存储的情况，扫描最坏 O(n log(K+1))，最终排序 O(K log(K+1))，辅助 O(K)。`top_k` 在 K 大约达到 n/4 时改用更适合大 K 的路线：Python、Java 全排序，C++ `nth_element` 后排序前 K 项。保留输入会增加 O(n) 副本；阈值是本次工作负载下的启发式，可以按部署环境重新测定。

这些实现强调清晰的不变量和可测量的改进。Python 标准库排序在本次测试中明显快于 Python 层手写归并；算法复杂度相同不能抵消解释器与成熟原生实现的差异。完整数字、比较口径及未覆盖的分布见[性能报告](BENCHMARKS.md)。
