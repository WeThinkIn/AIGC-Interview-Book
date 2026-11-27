# 实现参考与版本

核验日期：2026-11-27。这里记录实际阅读的固定版本与用途。三语言算法为独立实现，没有直接复制这些项目的代码；公开算法思想、标准库接口与工程反例用于审阅和对照。许可证描述对应参考材料，不替代其完整条款，也不将不同项目的许可证合并成一个授权。

<div align="center">

| 项目与固定版本 | 所读许可 | 借鉴范围 |
|---|---|---|
| [cp-algorithms/cp-algorithms · 9392ad98aaba](https://github.com/cp-algorithms/cp-algorithms/tree/9392ad98aabadd2bc2feab65b91881918dfe560e) | CC BY-SA 4.0（文章仓库） | 前缀函数、Dijkstra 及数据结构不变量；正文独立表述 |
| [kth-competitive-programming/kactl · a699a7a26a2e](https://github.com/kth-competitive-programming/kactl/tree/a699a7a26a2e61d0fdac07fc3f062c7661420009) | 所读 KMP.h 声明 CC0；其他文件按各自声明 | KMP 接口与哨兵假设；本库独立扫描，不依赖特殊符号 |
| [TheAlgorithms/Python · 30507bbe4139](https://github.com/TheAlgorithms/Python/tree/30507bbe41397065728167fb09aa820f9fce8dc7) | MIT | 归并教学对照；所读版本使用头部 pop，本库使用下标与缓冲 |
| [TheAlgorithms/C-Plus-Plus · fdcc29b0ac28](https://github.com/TheAlgorithms/C-Plus-Plus/tree/fdcc29b0ac2848e9d1a8d926e171cc5b39f97759) | MIT | 归并、边界与容器用法 |
| [TheAlgorithms/Java · 15a4a9d36387](https://github.com/TheAlgorithms/Java/tree/15a4a9d36387782eb324416984b6e465d1c3f7cc) | MIT | 归并和 Java 接口设计 |
| [abseil/abseil-cpp · d1cbe4fe0ecf](https://github.com/abseil/abseil-cpp/tree/d1cbe4fe0ecf25b6bd8e26c7cb2a98e32d5fc896) | Apache-2.0 | Swiss Tables 控制信息、分组探测及测试 |
| [facebookresearch/faiss · b14b3d693161](https://github.com/facebookresearch/faiss/tree/b14b3d693161c7b1cbc5f2e398c8234b42b04c8b) | MIT | IVF、复合索引、RaBitQ 接口及测试；用于问答 |
| [nmslib/hnswlib · ca426729609b](https://github.com/nmslib/hnswlib/tree/ca426729609b3221563047ceb269cc1213e0d376) | Apache-2.0 | 过滤接口、同条件 brute-force 测试 |
| [microsoft/DiskANN · 18da85209728](https://github.com/microsoft/DiskANN/tree/18da85209728f7dffd07f6f9c79fd2f0e42bd3b1) | MIT（LICENSE.txt） | README 的过滤接口与图索引；未采用失效文档 |
| [RoaringBitmap/RoaringBitmap · 0f02d33a8bba](https://github.com/RoaringBitmap/RoaringBitmap/tree/0f02d33a8bba78a3f6bbaef921721ae79d91e1e7) | Apache-2.0 | 分块容器、交并、退化分布与 run 优化 |
| [apache/datasketches-java · 028462110b0a](https://github.com/apache/datasketches-java/tree/028462110b0a995dcf49caecd3c59cbaf49ccea0) | Apache-2.0 | KLL 的秩误差、合并、参数与测试 |
| [sgl-project/sglang · 2cb7f6c230cd](https://github.com/sgl-project/sglang/tree/2cb7f6c230cde9d9961a7c44d55391b79ea3d671) | Apache-2.0 | Radix cache 匹配、拆边、引用与回收 |

</div>

固定版本避免把后续主分支变化混入判断。项目的关注度不保证每个教学示例都适合高性能路径。Python 列表头部删除、Java 装箱和 C++ 迭代器所有权，都需要落实到具体代码检查。

## 直接阅读入口

- [Swiss Tables](https://abseil.io/about/design/swisstables)：控制字节、H1/H2 与候选筛选。
- [vLLM 前缀缓存](https://docs.vllm.ai/en/latest/design/prefix_caching/)：父块身份、完整块与缓存域。
- [SGLang RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)：2024 年机制文章；不将当年吞吐数字当作当前保证。
- [DataSketches KLL](https://datasketches.apache.org/docs/KLL/KLLSketch.html)：压缩与秩误差。
- [Faiss 索引选择](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)：精确检索、分区、图与压缩。
- [LAPACK Linear Equations](https://netlib.org/lapack/lug/node38.html)：分解、条件估计、误差界与迭代改进。
- [Unicode UAX #29](https://www.unicode.org/reports/tr29/)：码点与文本分段。
- [稳健几何谓词](https://www.cs.cmu.edu/~quake/robust.html)：近退化情况下的方向判定。

工程接口存在与闭源模型采用某方法是不同证据。正文只采用公开代码能支持的机制，不从相似输出推断未公开架构，也不复述未经复现的加速倍数。
