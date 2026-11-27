# 性能基准与解释

核验日期：2026-11-27。数字来自重复进程的本机实验，比较相同语言内的相同任务，支持下面的实现取舍；不能据此推断跨硬件、任意输入或全球最快。

## 计时口径

- Intel Core i5-1038NG7，x86_64，macOS；CPython 3.11.13、Apple clang 14.0.0、Temurin JDK 21.0.13.1+2。
- C++：`-std=c++20 -O3 -DNDEBUG -Wall -Wextra -Werror`；Java：`-Xms256m -Xmx512m`。无 CPU 绑定，未隔离系统负载。
- n=32,768；同一 uint32 LCG，种子 20261127，递推 `1664525*x+1013904223` 按 32 位回绕，值取模 100000。
- 排序测随机、有序、逆序、重复 8 个值；Top K 测 K=8、128、16384；窗口最大值测 k=128。后两者仅测该随机序列，未覆盖全部退化分布。
- 每语言 3 个独立进程，每方法每进程 11 样本，共 33 次；Python/C++ 预热 8 次，Java 预热 40 次。
- 计入保持输入不变所需拷贝及结果分配；不计数据生成、编译、启动和 JSON 输出。计时前完整核对输出，结果在区间外消费。
- 单位毫秒。P10/P90 为排序样本的离散位置，表示波动，不是线上 P99 或置信区间。

## 实测结果

方法名与源码一致。merge_buffers 是本库归并，adaptive 是当前 Top K 策略，其余保留标准库或朴素对照。

### python

<div align="center">

| 任务 | 方法 | 中位数 ms | P10 ms | P90 ms |
|---|---|---|---|---|
| sort_random | merge_buffers | 59.878002 | 58.469023 | 61.974768 |
| sort_random | builtin_sorted | 4.729055 | 4.606687 | 5.006349 |
| sort_sorted | merge_buffers | 53.991508 | 51.932848 | 56.987936 |
| sort_sorted | builtin_sorted | 0.274397 | 0.263993 | 0.314226 |
| sort_reversed | merge_buffers | 56.965021 | 54.568070 | 60.225167 |
| sort_reversed | builtin_sorted | 0.943192 | 0.896536 | 1.318738 |
| sort_duplicates | merge_buffers | 53.471739 | 52.914180 | 54.838670 |
| sort_duplicates | builtin_sorted | 0.907795 | 0.888072 | 1.045769 |
| top_8 | bounded_heap | 0.592283 | 0.565365 | 0.696825 |
| top_8 | adaptive | 0.596218 | 0.568860 | 0.718958 |
| top_8 | heapq_nlargest | 0.565231 | 0.558138 | 0.664991 |
| top_8 | full_sort | 4.673789 | 4.526276 | 4.871493 |
| top_128 | bounded_heap | 0.722842 | 0.691825 | 0.914556 |
| top_128 | adaptive | 0.722973 | 0.689225 | 0.872966 |
| top_128 | heapq_nlargest | 0.829062 | 0.803868 | 1.070894 |
| top_128 | full_sort | 4.751153 | 4.555440 | 5.273712 |
| top_16384 | bounded_heap | 7.405205 | 7.110880 | 7.567034 |
| top_16384 | adaptive | 4.929178 | 4.568349 | 5.096077 |
| top_16384 | heapq_nlargest | 18.335764 | 17.748821 | 19.248438 |
| top_16384 | full_sort | 4.810591 | 4.625457 | 5.009738 |
| window_128 | monotonic_deque | 7.742397 | 7.518692 | 8.122709 |
| window_128 | scan_each_window | 59.004352 | 58.595011 | 59.375784 |

</div>

### cpp

<div align="center">

| 任务 | 方法 | 中位数 ms | P10 ms | P90 ms |
|---|---|---|---|---|
| sort_random | merge_buffers | 2.477600 | 2.326660 | 2.701240 |
| sort_random | std_stable_sort | 2.157620 | 2.010610 | 2.324660 |
| sort_sorted | merge_buffers | 0.458637 | 0.450878 | 0.518456 |
| sort_sorted | std_stable_sort | 0.132128 | 0.131737 | 0.139007 |
| sort_reversed | merge_buffers | 0.560797 | 0.534605 | 0.739395 |
| sort_reversed | std_stable_sort | 1.452480 | 1.400790 | 1.628940 |
| sort_duplicates | merge_buffers | 0.549564 | 0.539226 | 0.688676 |
| sort_duplicates | std_stable_sort | 0.891492 | 0.864688 | 1.046400 |
| top_8 | bounded_heap | 0.012144 | 0.011986 | 0.012753 |
| top_8 | adaptive | 0.012083 | 0.011904 | 0.013291 |
| top_8 | priority_queue | 0.028665 | 0.028463 | 0.030254 |
| top_8 | nth_then_sort | 0.112836 | 0.100167 | 0.169331 |
| top_8 | full_sort | 1.753570 | 1.683760 | 1.972600 |
| top_128 | bounded_heap | 0.030308 | 0.027855 | 0.033134 |
| top_128 | adaptive | 0.027675 | 0.026762 | 0.030457 |
| top_128 | priority_queue | 0.048765 | 0.047753 | 0.049462 |
| top_128 | nth_then_sort | 0.114374 | 0.104304 | 0.136550 |
| top_128 | full_sort | 1.726390 | 1.678330 | 1.912260 |
| top_16384 | bounded_heap | 1.704070 | 1.679130 | 1.917500 |
| top_16384 | adaptive | 0.983953 | 0.961071 | 1.160260 |
| top_16384 | priority_queue | 1.740540 | 1.694080 | 1.954550 |
| top_16384 | nth_then_sort | 0.989470 | 0.960432 | 1.107890 |
| top_16384 | full_sort | 1.697520 | 1.676400 | 1.910110 |
| window_128 | monotonic_ring | 0.515533 | 0.509221 | 0.602479 |
| window_128 | scan_each_window | 2.896960 | 2.767550 | 3.104670 |

</div>

### java

<div align="center">

| 任务 | 方法 | 中位数 ms | P10 ms | P90 ms |
|---|---|---|---|---|
| sort_random | merge_buffers | 3.069085 | 2.867102 | 3.258858 |
| sort_random | arrays_sort | 2.217458 | 2.084312 | 2.425325 |
| sort_sorted | merge_buffers | 0.826605 | 0.756848 | 1.066198 |
| sort_sorted | arrays_sort | 0.214550 | 0.195576 | 0.270673 |
| sort_reversed | merge_buffers | 1.165299 | 1.031116 | 1.310822 |
| sort_reversed | arrays_sort | 0.221197 | 0.204790 | 0.270606 |
| sort_duplicates | merge_buffers | 1.125217 | 1.064453 | 1.295670 |
| sort_duplicates | arrays_sort | 0.258812 | 0.173490 | 0.366051 |
| top_8 | primitive_heap | 0.093603 | 0.093288 | 0.094804 |
| top_8 | adaptive | 0.065379 | 0.063427 | 0.071344 |
| top_8 | priority_queue | 0.198011 | 0.189994 | 0.230485 |
| top_8 | full_sort | 2.004481 | 1.959284 | 2.254215 |
| top_128 | primitive_heap | 0.043306 | 0.041902 | 0.046492 |
| top_128 | adaptive | 0.042571 | 0.041387 | 0.044485 |
| top_128 | priority_queue | 0.261158 | 0.243860 | 0.304249 |
| top_128 | full_sort | 2.041417 | 1.965048 | 2.211588 |
| top_16384 | primitive_heap | 2.216040 | 2.017040 | 2.451406 |
| top_16384 | adaptive | 2.271263 | 1.996959 | 2.399286 |
| top_16384 | priority_queue | 5.666696 | 5.391730 | 5.942602 |
| top_16384 | full_sort | 2.109984 | 2.012658 | 2.390514 |
| window_128 | monotonic_ring | 0.706138 | 0.677667 | 0.918459 |
| window_128 | scan_each_window | 3.044209 | 2.850801 | 3.227575 |

</div>

## 结果解释

Python 随机排序中，归并约 59.88 ms，sorted 约 4.73 ms。归并规则正确，但成熟原生排序明显更快。C++ 对照为 std::stable_sort。Java 使用 long[]，相等整数的记录身份不可观察；若改为对象记录，应重新检查稳定性与吞吐。

K=128 的 Java 基本类型堆约 0.0433 ms，装箱 PriorityQueue 约 0.2612 ms，说明该热循环的表示与更新方式有价值。Python 堆方法间差距小得多，需结合波动判断。

K=n/2 时，Python adaptive 约 4.93 ms、堆约 7.41 ms；C++ 分区后排序约 0.984 ms、堆约 1.704 ms。这支持大 K 保留另一条路线。Java adaptive 约 2.27 ms、基本类型堆约 2.22 ms，区间重叠，本轮没有稳定胜出的证据。n/4 阈值没有经过密集扫描，只是可调策略；内存敏感或流式任务可明确选择 heap 入口。

单调队列相对逐窗口重扫，改进来自每个下标至多进出一次。没有对 LRU、Trie、最短路或数值算法做性能排名，也未测试 GPU、并行线程或 NUMA。

## 复现与检查

运行命令见 [README](README.md)。[汇总 JSON](evidence/benchmark_20261127.json)包含环境、源码哈希和 69 行结果；同目录按语言与进程保存全部 JSONL 样本。[正确性报告](evidence/verification_20261127.json)记录 3,407 组用例。测试说明在这些输入与契约下未发现错误，不是对全部输入的形式化证明。

改变规模、K/N、分布或平台后应重新测量。确定阈值需要扫描代表性分布并用独立数据验证；评估线上收益则需放回端到端请求，检查延迟与内存是否改善。
