# 《三年面试五年模拟》之模型部署基础知识高频考点（精华版）

> **4 个章节 · 48 个大问题 · 128 道面试题**

本模块系统梳理模型部署方向的高频面试考点，整体按「**综述 → 传统模型 → 大模型 → 性能调优**」的主线组织：先建立推理部署的全局认知框架，再分别深入 ONNX/TensorRT 的传统模型部署链路与大模型推理框架（vLLM / SGLang）的核心技术，最后落地到性能分析与调优的工具链和方法论。

每道面试题均标注 **难度评分** 与 **考察频率**（⭐ 1~5 分），可按优先级安排复习。

## 章节导航

- :star: **[6.1 推理部署综述](6.1推理部署综述.md)** — 18 个大问题 · 44 道面试题

  从推理框架的定义、核心功能与分类选型出发，讲清“为什么需要推理加速”以及影响延迟/吞吐的关键因素；进而覆盖模型压缩三大路线（剪枝、蒸馏、量化）、模型编译与 JIT、算子融合与内核优化、计算强度与 Roofline 模型、内存优化、静态/动态/连续批处理、CPU/GPU/NPU 硬件选型、Docker 与服务化部署，以及推理服务的性能与稳定性评估。

- :blue_book: **[6.2 传统模型部署：ONNX与TensorRT](6.2传统模型部署：ONNX与TensorRT.md)** — 8 个大问题 · 24 道面试题

  以 ONNX 与 TensorRT 两条主线展开：ONNX 的 Graph/Node/Tensor/Opset 组成、中间表示的多方对比、PyTorch 导出流程（dynamic_axes、opset_version）与实战排错（算子不支持、动态 Shape、精度不一致）；ONNX Runtime 的 Execution Provider 与图优化；TensorRT 的加速原理、从 ONNX 到 Engine 的构建流程、动态 Shape 调优与 FP16/INT8 精度评估；最后是 Triton Inference Server 的模型仓库、Backend、Instance Group 与动态批处理。

- :rocket: **[6.3 大模型部署技术](6.3大模型部署技术.md)** — 10 个大问题 · 32 道面试题

  从大模型推理与传统推理的本质差异（自回归解码、Prefill/Decode 两阶段、KV Cache 显存墙）切入，深入 vLLM V1（EngineCore、PagedAttention、连续批处理、Prefix Caching）与 SGLang（RadixAttention、前端 DSL、约束解码）；覆盖大模型量化（PTQ/QAT/AWQ/GPTQ/FP8、KV Cache 量化）与底层优化（FlashAttention、MQA/GQA、CUDA Graph）；再延伸至 PD 分离、推测解码、TTFT/TPOT 评测与压测、分布式并行部署（TP/PP/DP），以及显存 OOM、延迟抖动等生产环境排查。

- :eyes: **[6.4 性能分析与调优工具](6.4性能分析与调优工具.md)** — 12 个大问题 · 28 道面试题

  从性能目标、指标体系与 SLO 基线出发，梳理在线/离线剖析与系统级/内核级工具的选型配合；逐一讲解 PyTorch Profiler、Nsight Systems、Nsight Compute、trtexec、ONNX Runtime Profiler 与 NVTX 的使用与解读；深入全链路时耗拆解、Compute-Bound 与 Memory-Bound 判别、GPU 利用率与 SM Occupancy 分析、Kernel Launch Overhead 与 CPU-GPU 同步点定位；最后给出大模型推理专项分析（TTFT/TPOT、KV Cache 命中率、Continuous Batching 调度效率）与 Profile → Analyze → Hypothesize → Optimize → Verify 的闭环调优方法论。

## 建议阅读顺序

1. **建立框架** — 先读 6.1，理解推理部署的全景与关键影响因素，后续两章的优化手段都能在这里找到“为什么要做”的动机。
2. **分方向深入** — 面试偏 CV/传统模型岗位重点看 6.2；大模型相关岗位重点看 6.3。两章中的量化、算子融合、批处理等知识点与 6.1 相互呼应。
3. **掌握调优** — 6.4 是前面所有优化手段的“验证环节”，工具链与方法论在面试中常作为“你如何定位性能问题”的答题框架使用。
