# 目录

## 第一章 Stable Diffusion 系列核心高频考点

[1.介绍一下Stable Diffusion的原理](#q-028)
  - [面试问题：Stable Diffusion 相比经典 Diffusion model 的核心优化是什么？](#q-029)
  - [面试问题：介绍一下 Stable Diffusion 的训练 / 推理过程（正向扩散过程和反向去噪过程）](#q-030)
  - [面试问题：Stable Diffusion 训练时为什么要为每个样本随机采样一个时间步？该采样策略对模型质量有什么影响？](#q-031)
  - [面试问题：Stable Diffusion 中的 ε-prediction、x0-prediction、v-prediction 三种参数化方式有何差异？SD 各版本分别采用了哪种？为什么？](#q-031a)
  - [面试问题：Stable Diffusion 中的 latent scale factor（如 0.18215）有什么作用？为什么不同 SD 版本的 scale factor 不同？](#q-032a)
  - [面试问题：介绍一下针对 Stable Diffusion 的模型融合技术](#q-033)
  - [面试问题：Stable Diffusion 进行模型融合的技巧有哪些？](#q-034)
  - [面试问题：Stable Diffusion 中是如何添加时间步 timestep 信息的？](#q-035)
  - [面试问题：Stable Diffusion 训练时需要设置 timesteps=1000，为什么推理时只用几十步就可以生成图片？](#q-036)
  - [面试问题：为什么相同 seed + 相同 prompt 在不同采样器 / 精度 / 框架下结果会有差异？工程上如何保证生成结果可复现？](#q-036a)
  - [面试问题：Stable Diffusion 中的 negative prompt（反向提示词）是如何加入的？](#q-037)
  - [面试问题：Stable Diffusion 中文本条件是如何一步步控制图像生成的？请完整描述从 Prompt 到 Latent 的注入链路](#q-038)
  - [面试问题：介绍 Stable Diffusion 核心网络结构](#q-039)
  - [面试问题：Stable Diffusion 中的 Inpaint 和 Outpaint 分别是什么？](#q-040)
  - [面试问题：Stable Diffusion 中的 img2img（图生图）原理是什么？denoising strength 起到什么作用？](#q-040a)
  - [面试问题：Stable Diffusion 训练 / 推理为什么需要 EMA（指数滑动平均）权重？常见 EMA decay 的取值与权衡是什么？](#q-040b)
  - [面试问题：Stable Diffusion 系列在 fp32 / fp16 / bf16 / fp8 下的推理质量与显存差异是什么？工业上如何选择推理精度？](#q-040c)

[2.介绍一下 Stable Diffusion 中 VAE 的架构、原理和作用](#q-041)
  - [面试问题：VAE 为什么会导致图像变模糊？](#q-042)
  - [面试问题：为什么 VAE 单独做生成效果不好，但是 VAE + Diffusion 的图像生成效果就很好？](#q-043)
  - [面试问题：Stable Diffusion 模型中的 VAE 和单纯的 VAE 生成模型的区别是什么？](#q-044)
  - [面试问题：从 SD 1.x → SDXL → SD 3 → FLUX.1，VAE 在通道数、下采样率、训练目标上的演进路线是怎样的？](#q-044a)
  - [面试问题：VAE 编码后为什么要乘以 scale_factor？SD 各版本的 scale_factor 是如何确定的？](#q-044b)
  - [面试问题：SDXL VAE 在 fp16 下出现"白图 / NaN"问题的原因是什么？工业上常见的修复方案有哪些？](#q-044c)
  - [面试问题：大分辨率推理时如何降低 VAE 解码显存？VAE Tiling 与 TAESD 各自的取舍是什么？](#q-044d)
  - [面试问题：VAE / Tokenizer / Latent 空间为什么会影响图像生成质量和训练效率？](#q-044e)

[3.介绍一下 Stable Diffusion 中 Backbone 的架构、原理和作用](#q-045)
  - [面试问题：Stable Diffusion 是如何在 U-Net 内部把文本与图像两种模态的语义对齐的？](#q-046)
  - [面试问题：介绍一下 Stable Diffusion 中的交叉注意力机制](#q-047)
  - [面试问题：Stable Diffusion 中 cross attention 的 Q / K / V 分别是什么？为什么图像隐变量作为 Q，文本 Prompt 作为 K / V？](#q-048)
  - [面试问题：为什么使用 U-Net 作为 Stable Diffusion 模型的核心架构？介绍一下 U-Net 架构](#q-049)
  - [面试问题：为什么 SD U-Net 中 Self-Attention 与 Cross-Attention 主要放在中、低分辨率层？高分辨率层为何以卷积为主？](#q-049a)
  - [面试问题：U-Net 与 DiT / MM-DiT 在 Backbone 设计哲学上的本质差异是什么？SD 系列从 U-Net 演进到 DiT 的根本原因是什么？](#q-049b)
  - [面试问题：Stable Diffusion 中常见的注意力加速技术（xFormers、SDPA、FlashAttention、Token Merging / ToMe）的核心思想与适用场景是什么？](#q-049c)
  - [面试问题：SD Backbone 中 GroupNorm + SiLU + 残差连接的设计为何对训练稳定性很关键？换成 LayerNorm / BatchNorm 会有什么问题？](#q-049d)

[4.介绍一下 Stable Diffusion 中 Text Encoder 的架构、原理和作用](#q-050)
  - [面试问题：举例介绍一下 Stable Diffusion 模型进行文本编码的全过程](#q-051)
  - [面试问题：Stable Diffusion 如何通过文本来实现对图像生成内容的控制？SD 中是如何注入文本信息的？](#q-052)
  - [面试问题：Negative Prompt 实现的原理是什么？](#q-053)
  - [面试问题：CLIP Text Encoder 的 77 tokens 长度限制对长 Prompt 的实际影响是什么？工程上如何突破（chunking、weighted prompt、T5 等长上下文编码器）？](#q-053a)
  - [面试问题：Prompt 中的权重语法（(word:1.2)、[word]）的实现原理是什么？A1111 / ComfyUI / Compel 三种 Prompt 解析方式有何差异？](#q-053b)
  - [面试问题：CLIP skip 是什么？为什么社区在 SD 1.5 上常用 clip_skip=2，但 SDXL / SD 3 不再推荐？](#q-053c)
  - [面试问题：为什么 SD 1.x 选用 CLIP ViT-L 而 SD 2.x 切换为 OpenCLIP ViT-H？这一切换给生成效果带来了哪些可观察的差异？](#q-053d)
  - [面试问题：Text Encoder 和 VLM 条件编码器在图像生成模型中起什么作用？](#q-053e)
  - [面试问题：如何处理 Prompt 和生成的图像不对齐的问题？](#q-054)
  - [面试问题：扩散模型是如何引入控制条件的？](#q-055)

[5.Stable Diffusion XL 有哪些创新点？](#q-056)
  - [面试问题：与 Stable Diffusion 相比，Stable Diffusion XL 的核心优化有哪些？](#q-057)
  - [面试问题：Stable Diffusion XL 的 VAE 部分有哪些创新？详细分析改进意图](#q-058)
  - [面试问题：Stable Diffusion XL 的 Backbone 部分有哪些创新？详细分析改进意图](#q-059)
  - [面试问题：Stable Diffusion XL 的 Text Encoder 部分有哪些创新？详细分析改进意图](#q-060)
  - [面试问题：Stable Diffusion XL 中使用的训练方法有哪些创新点？](#q-061)
  - [面试问题：SDXL 的 micro-conditioning（original_size / crop_top_left / target_size）是什么？为什么是 SDXL 工程化层面最关键的创新之一？](#q-061a)
  - [面试问题：SDXL 的双 Text Encoder + Pooled Embedding 注入机制具体是怎样的？工程实现中有哪些容易忽略的细节？](#q-061b)
  - [面试问题：训练 Stable Diffusion XL 时为什么要使用 offset Noise？](#q-062)
  - [面试问题：介绍一下 Stable Diffusion XL Turbo 的原理](#q-063)
  - [面试问题：SDXL-Turbo 用的蒸馏方法是什么？](#q-064)
  - [面试问题：SDXL Lightning、DMD2、Hyper-SD 等新一代少步蒸馏方法相比 SDXL-Turbo 有哪些进步？](#q-064a)
  - [面试问题：什么是 SDXL Refiner？](#q-065)

[6.Stable Diffusion 3 有哪些创新点？](#q-066)
  - [面试问题：介绍一下 Stable Diffusion 3 的整体架构。与 Stable Diffusion XL 相比，SD 3 的核心架构优化有哪些？详细分析改进意图（VAE、Backbone、Text Encoder）](#q-067)
  - [面试问题：MM-DiT 与原始 DiT 的核心差异是什么？为什么 SD 3 选择 MM-DiT 而不是直接复用 DiT？](#q-067a)
  - [面试问题：SD 3 的 Rectified Flow 训练目标相比 ε-prediction 的本质差别是什么？给少步采样带来了哪些工程优势？](#q-067b)
  - [面试问题：Stable Diffusion 3 中使用的训练方法有哪些创新点？](#q-068)
  - [面试问题：SD 3 / SD 3.5 在高分辨率训练中对 timestep schedule 做的 shift 具体是怎么做的？为什么对大尺寸训练至关重要？](#q-068a)
  - [面试问题：训练 Stable Diffusion 过程中官方使用了哪些训练技巧？](#q-069)
  - [面试问题：介绍一下 Stable Diffusion 3.5 系列的原理](#q-070)
  - [面试问题：为什么 Stable Diffusion 3 使用三个文本编码器？](#q-071)
  - [面试问题：Stable Diffusion 3 中数据标签工程的具体流程是什么样的？](#q-072)
  - [面试问题：SD3-Turbo 用的蒸馏方法是什么？](#q-073)
  - [面试问题：Stable Diffusion 3 的图像特征和文本特征在训练前缓存策略有哪些优缺点？](#q-074)
  - [面试问题：Stable Diffusion 3.5 的改进在哪里？](#q-075)
  - [面试问题：SD 3 / SD 3.5 与 FLUX.1 在架构与训练范式上的继承与差异是什么？](#q-075a)


## 第二章 FLUX系列核心高频考点

[1.介绍一下FLUX.1的原理，与Stable Diffusion 3相比有哪些创新点？](#q-flux-001)

[2.FLUX.1在训练过程中使用了哪些优化技巧？](#q-flux-002)
  - [面试问题：FLUX.1模型的微调训练流程一般包含哪几部分核心内容？](#q-flux-003)
  - [面试问题：FLUX.1模型的微调训练流程中有哪些关键参数？](#q-flux-004)

[3.介绍一下FLUX.1 Lite与FLUX.1的异同](#q-flux-005)

[4.介绍一下FLUX.1 Kontext的原理，有哪些创新点？](#q-flux-006)
  - [面试问题：介绍一下FLUX.1 Kontext的原理，FLUX.1 Kontext能够执行哪些AIGC任务？](#q-flux-007)

[5.介绍一下FLUX.1 Krea的原理，有哪些创新点？](#q-flux-008)
  - [面试问题：介绍一下FLUX.1 Krea的训练策略](#q-flux-009)
  - [面试问题：介绍一下FLUX.1-Krea中监督微调（SFT）的流程](#q-flux-010)
  - [面试问题：FLUX.1 Krea的后训练过程中有哪些关键要点？](#q-flux-011)
  - [面试问题：介绍一下FLUX.1 Krea的后训练过程中使用的Tuned Preference Optimization技术](#q-flux-012)

[6.与FLUX.1相比，FLUX.2有哪些创新点？](#q-flux-013)


---

<h1 id="ch-02">第一章 Stable Diffusion 系列核心高频考点</h1>

<h1 id="q-028">1.介绍一下Stable Diffusion的原理</h1>

<h2 id="q-029">面试问题：Stable Diffusion 相比经典 Diffusion model 的核心优化是什么？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

Rocky认为我们可以将Latent Diffusion Models（LDM）当作是一个开创性的通用算法模型框架，而Stable Diffusion是在此框架基础上，通过一系列工程技术优化后形成的、在开源社区大规模落地应用的成熟算法技术即产品的模型产品。

Stable Diffusion对原始LDM框架的具体改进主要体现在以下几个方面：

**工程化与稳定性优化**

1. 训练稳定性：通过改进噪声调度、梯度裁剪等训练技巧，减少了训练过程中出现模式崩溃或不稳定的风险，使模型更容易在大规模数据集上收敛。
2. 推理速度：在继承潜在空间高效性的基础上，持续优化去噪采样器的效率，并出现了像DeepCache、ToMe-SD、Xformer等专为SD设计的加速技术，进一步提升生成速度。

**模型架构与能力的增强**

1. 条件控制：虽然LDM框架能够支持文本条件作为输入，但编码文本信息的部分是一个随机初始化的Transformer模型；而Stable Diffusion通过一个预训练好的CLIP Text Encoder来编码文本信息，预训练过的模型往往要优于从零开始训练的模型，这个优化极大地提升了文本到图像的生成能力和语义遵循度。后续更衍生出LoRA、ControlNet等辅助模型，实现了对生成内容（如构图、姿态）的精细控制。
2. 生成质量与分辨率：通过在更大规模高质量数据上训练（Latent Diffusion Model是采用laion-400M数据训练的，而Stable Diffusion是在laion-2B-en数据集上训练的），同时Stable Diffusion的训练分辨率也更大（Latent Diffusion Model只是在256x256分辨率上训练，而Stable Diffusion先在256x256分辨率上预训练，然后再在512x512分辨率上进行微调训练），以及架构的持续迭代（如SDXL、SD3、FLUX.1、FLUX.2等），在图像细节、光影和分辨率上不断突破。

**开源生态与易用性**

这是Stable Diffusion产生巨大影响的关键。其开源策略催生了如ComfyUI、AUTOMATIC1111 WebUI等图形化界面工具，让普通用户也能轻松使用。庞大的开源社区贡献了海量的定制化模型、风格LoRA和实用AI绘画插件，使其从一个模型演变成一个功能极其丰富的AIGC图像创作生态系统。

简单来说，两者的关系可以概括为：Latent Diffusion Models (LDM) 是奠定核心思想的“论文”与“蓝图”；而Stable Diffusion (SD) 则是基于这张蓝图建造出的、不断升级的“摩天大楼”及围绕它形成的“繁荣城市”。


<h2 id="q-030">面试问题：介绍一下 Stable Diffusion 的训练 / 推理过程（正向扩散过程和反向去噪过程）</h2>


<h2 id="q-031">面试问题：Stable Diffusion 训练时为什么要为每个样本随机采样一个时间步？该采样策略对模型质量有什么影响？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

Stable Diffusion 在每个训练 step 中，对一个 batch 内的每个样本**独立、均匀地**从 $\{1, 2, \dots, T\}$（通常 $T=1000$）中采样一个时间步 $t$，再用 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 一步加噪、预测噪声。这是 **Monte Carlo 估计变分下界（ELBO）** 的工程实现。

#### 1. 为什么要随机采样而不是顺序遍历

- **理论上**：DDPM 的训练损失是对所有时间步 $t$ 求期望 $\mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$，逐 step 随机采样是这个期望的无偏估计。
- **工程上**：若顺序遍历 $t$，模型在某段连续 step 内只学某个噪声水平，梯度方向被局部时间步主导，**优化方向震荡、收敛慢**；随机采样使 batch 内同时覆盖低、中、高噪声段，梯度方向更稳定。
- **数据高效**：同一张图在不同 epoch 中会被随机匹配到不同的 $t$，等价于做了**隐式的数据增强**。

#### 2. 采样策略对模型质量的影响

- **均匀采样（DDPM 默认）**：实现最简单，但中等噪声段对最终视觉质量贡献最大，均匀采样导致中等 $t$ 的样本利用率不够极致。
- **重要性采样 / Loss-aware sampling**（Improved DDPM、SD3）：根据每个 $t$ 的 loss 大小动态调整采样概率，把更多算力分配给「难学」的时间步，加速收敛。
- **Logit-Normal / lognorm shift**（SD3、FLUX 中的 Rectified Flow 训练）：把 $t$ 偏向中间区域采样，对 RF 训练目标更友好，能提升采样步数较少时的生成质量。
- **大分辨率训练时的 schedule shift**（SD3、SDXL 高分辨率训练）：高分辨率图像的「信息破坏速度」与 $t$ 不再线性，需要把 schedule 偏移到更高 $t$，否则会出现「加噪不足，残留低频结构」问题。

**面试金句**：随机采样是无偏估计 ELBO 的需要；而**采样分布的形状**（均匀 / 重要性 / lognorm / shift）则直接决定了模型在不同噪声段的学习预算，是 SD3、FLUX 这类新一代模型重点优化的工程细节。


<h2 id="q-031a">面试问题：Stable Diffusion 中的 ε-prediction、x0-prediction、v-prediction 三种参数化方式有何差异？SD 各版本分别采用了哪种？为什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

扩散模型在数学上等价的三种「网络要预测什么」的选择，但在**数值稳定性、信噪比覆盖、与采样器/CFG 的兼容性**上差异巨大，是 SD 系列代际演进的关键技术点。

#### 1. 三种参数化的数学定义

记加噪公式 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$，定义信噪比 $\text{SNR}(t) = \bar\alpha_t / (1-\bar\alpha_t)$。三种预测目标的关系为：

```math
v_t = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0
```

```math
\epsilon = \sqrt{\bar\alpha_t}\,v_t + \sqrt{1-\bar\alpha_t}\,x_t,\quad
x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,v_t
```

#### 2. 三者对比

<div align="center">

| 预测目标 | 损失主导区间 | 高 $t$（接近纯噪声） | 低 $t$（接近原图） | 适用场景 |
| --- | --- | --- | --- | --- |
| **ε-pred** | 中、高噪声段 | 良好（噪声有信号） | 数值不稳定（信号占比小，loss 趋零） | 标准 DDPM、SD 1.x、SD 2.0、SDXL base |
| **x0-pred** | 低、中噪声段 | 数值不稳定（基本是噪声） | 良好 | 教师蒸馏、Inpainting 微调 |
| **v-pred** | 全噪声段均衡 | 良好 | 良好 | SD 2.1-v、SDXL 部分 fine-tune、Imagen、Rectified Flow |

</div>

#### 3. SD 各版本的选择

- **SD 1.x、SD 2.0**：ε-prediction，沿用 DDPM 原始范式。
- **SD 2.1-v（768 模型）**：v-prediction。Stability 官方在 768 高分辨率模型上切换到 v-pred，原因是高分辨率训练中**低 $t$ 区域 ε 数值非常小，loss 几乎为零**，模型学不到细节修复能力；v-pred 在所有 $t$ 上 loss 量级均衡，训练更稳定，CFG 也更不容易过曝。
- **SDXL base**：仍用 ε-pred（向下兼容生态），但 SDXL 的部分官方 / 社区微调版本（如 `sdxl-vpred`、`zsnr` 配方）使用 v-pred + Zero-SNR 终端噪声。
- **SD 3 / FLUX**：Rectified Flow 在数学上等价于 **v-prediction 的连续时间形态**——网络预测「从噪声到数据的速度场」，本质上把 v-pred 的全局均衡性发挥到极致，再叠加直线化路径以加速采样。

**面试金句**：三种参数化在数学上等价但在数值上不等价；**ε-pred 偏好高 $t$，x0-pred 偏好低 $t$，v-pred 在全 $t$ 均衡**。SD 系列从 1.x 的 ε-pred → 2.1-v 的 v-pred → SD 3 / FLUX 的 Rectified Flow，本质上是「让网络在所有噪声水平上都得到均衡的梯度信号」这条路线的不断深化。


<h2 id="q-032a">面试问题：Stable Diffusion 中的 latent scale factor（如 0.18215）有什么作用？为什么不同 SD 版本的 scale factor 不同？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

`scale_factor` 是把 VAE Encoder 输出的 latent 喂给扩散模型之前，要乘以的一个标量常数；推理时 VAE Decoder 之前再除回去。它的核心作用是：**让 latent 的统计分布近似单位方差的标准正态**，从而与扩散模型的噪声 schedule 相匹配。

#### 1. scale factor 的作用

- **统计对齐**：扩散模型默认假设输入分布近似 $\mathcal{N}(0, I)$（前向加噪、反向去噪都基于这个假设）。VAE Encoder 训练时只优化重建质量，并未约束输出 latent 的方差恰好为 1；如果不缩放，latent 的方差可能远大于或远小于 1，导致：
  - 加噪过程把信号「淹没」过快或过慢；
  - 同一 noise schedule 下信噪比错位，CFG / 采样器表现劣化。
- **数值稳定**：把 latent 拉回 $\mathcal{O}(1)$ 量级有利于 fp16 / bf16 的数值范围。
- **与已发布权重耦合**：scale factor 和扩散网络是**一体训练**的，所以推理时必须用与训练完全一致的常数，否则结果会整体偏色或塌缩。

#### 2. 为什么不同版本 scale factor 不同

`scale_factor` 不是手工调出的「魔法数字」，而是按 **「在训练数据集上让 latent 的标准差近似 1」** 这个原则**统计估计**出来的：把 VAE 跑在大批训练图上，估出 latent 的 std，取倒数即为 scale factor。

<div align="center">

| 版本 | VAE 通道数 | scale_factor | 备注 |
| --- | --- | --- | --- |
| SD 1.x / 2.x | 4 | **0.18215** | 在 LAION 子集上估计的 latent std≈5.49 的倒数 |
| SDXL | 4 | **0.13025** | SDXL 重新训练了 VAE，latent 分布发生变化 |
| SD 3 / FLUX | 16 | 由 `scaling_factor` + `shift_factor` 联合定义 | 16 通道 VAE 同时引入 mean shift，latent 先减 shift 再乘 scale |

</div>

#### 3. 工程注意事项

- **跨版本切换 VAE 必须同步 scale_factor**：把 SD 1.5 的 VAE 直接用到 SDXL 上、不改 scale factor，会导致明显偏色或细节崩溃。
- **SD 3 / FLUX 的 latent 是「先减 shift 后乘 scale」**：忽略 shift 项是迁移代码时的高频踩坑点。
- **diffusers / ComfyUI 中**这个常数通常已经写在 `vae.config.scaling_factor` 中，自定义 pipeline 时必须读取而不是硬编码。

**面试金句**：scale factor 的本质是把「重建友好的 VAE 隐空间」对齐到「扩散友好的单位方差正态空间」；它和扩散网络是绑定训练的一对常数，跨版本/跨 VAE 必须同步切换。


<h2 id="q-033">面试问题：介绍一下针对Stable Diffusion的模型融合技术</h2>

Stable Diffusion的模型融合主要通过 **Merge Block Weight（块权重融合）** 这种精细化的模型参数整合技术实现，通过分层处理U-Net/Transformer内部不同功能模块层的权重，实现多个Stable Diffusion模型特点优势的定向组合。

#### 一、核心原理：分层权重插值

模型融合的目标是合并多个训练好的Stable Diffusion模型（如风格模型+主体模型），生成兼具各方优势的新模型。Merge Block Weight的核心创新在于**分块处理U-Net/Transformer结构**，而非整体融合：

**1. U-Net结构解构**

Stable Diffusion的U-Net包含多个功能模块：

- **ResBlock**：负责基础特征提取与残差连接
- **Spatial Transformer（Cross-Attention）**：融合文本与图像语义
- **DownSample/UpSample**：控制特征图分辨率变换

**2. 分块独立融合**

对每个模块的权重独立计算插值，公式为：

```math
W_{\text{merged}}^{(i)} = \alpha \cdot W_A^{(i)} + (1 - \alpha) \cdot W_B^{(i)}
```

其中 $W_A^{(i)}$ 和 $W_B^{(i)}$ 是待融合模型在模块 $i$ 的权重， $\alpha$ 为该模块的融合系数（0~1）。

#### 二、技术实现流程

##### 1. 权重归一化（关键预处理）

- 目的：解决不同模型参数分布差异导致的融合冲突
- 方法：对每个模型的权重进行LayerNorm或Min-Max缩放，使其处于相近数值范围

##### 2. 插值算法选择

<div align="center">

| **算法** | 适用场景 | 优势 | 缺点 |
|----------|----------|------|------|
| **线性插值（LERP）** | 简单融合、硬件资源有限 | 计算效率高 | 可能丢失非线性特征 |
| **球面线性插值（SLERP）** | 高质量风格融合（如艺术风格） | 保持权重向量方向一致性，避免特征坍缩 | 计算复杂度高 |

</div>

##### 3. 分层系数配置

不同模块需设置差异化融合系数，例如：

- **ResBlock**： $\alpha=0.5$ （平衡底层特征）
- **Spatial Transformer**： $\alpha=0.8$ （侧重模型A的文本控制力）
- **UpSample层**： $\alpha=0.3$ （侧重模型B的细节生成能力）

#### 总结

Merge Block Weight通过解构U-Net并分层融合权重，实现了模型能力的精准嫁接，成为解决单一模型局限性问题的关键技术。随着Stable Diffusion 3等新架构对多模态权重的分离设计（如MMDiT），模型融合将进一步向**模态感知融合**（Modality-Aware Merging）演进，在艺术创作、工业设计等领域释放更大潜力。

<h2 id="q-034">面试问题：Stable Diffusion进行模型融合的技巧有哪些？</h2>

我们在进行几个Stable Diffusion的融合时，可以调整U-Net架构中每一层模型的融合权重，从而能够进行模型融合的进阶整合：

在MBW插件中，将U-Net分层了25个可调层，开源社区将其分为:
IN区：有12层
M区：有1层
OUT区：有12层

IN区影响下采样过程对特征的提取，层数从00到11，感受野越来越大，影响的程度越来越大。IN区块负责平面构成的相关工作（构图元素以及生成图像背景），特别是6-11层，总的来说层数越高影响效果越明显，更改层数越多影响效果越明显。比如：各个物体的大小、位置以及基本轮廓。其中在画面中占比越小的物体受到越浅层的参数控制，占比大的物体受到更深层的参数控制。浅层权重越高，小物体的表现效果就越向该模型靠拢；深层权重越高，较大物体的表现效果就越向该模型靠拢。

OUT区影响上采样过程对特征进行还原，层数从00到11，感受野越来越小，影响的程度越来越小。OUT区块负责色彩构成和画风的相关工作，主要是0-4层起核心作用，同时如果是人物图像，2-7层可以控制脸部的微调。

在IN区中编号越高，对平面构成的影响就越偏向大体。在OUT区中，编号越高，对细化过程的影响就越局域化，对上色过程的影响就越大体化。比如：基本色调，色彩丰富或单一，皮肤质感，光影，线条。深层参数负责大区域的色彩，比如基本色调、色彩丰富度与光影；浅层参数负责细节的色彩，比如线条是否清晰，通过浅层可以调整手指；深层与浅层之间的中层则负责区域的色彩，区域内色彩的不同变化程度可以体现出不同的皮肤质感和区域光影效果。

同时如果IN层和OUT层只改变其中的某一层，几乎不会产生影响效果。

M区：影响最大的一层，甚至比IN11层的影响更大，起到了类似IN层的作用，可以看作IN12层，但也只能起到一层的作用，不如IN层中多层叠加后的影响大。该层越大，构图越向该模型靠拢。


<h2 id="q-035">面试问题：Stable Diffusion 中是如何添加时间步 timestep 信息的？</h2>


<h2 id="q-036">面试问题：Stable Diffusion 训练时需要设置 timesteps=1000，为什么推理时只用几十步就可以生成图片？</h2>

目前扩散模型训练一般使用DDPM（Denoising Diffusion Probabilistic Models）采样方法，但推理时可以使用DDIM（Denoising Diffusion Implicit Models）采样方法，DDIM通过去马尔可夫化，大大减少了扩散模型在推理时的步数。


<h2 id="q-036a">面试问题：为什么相同 seed + 相同 prompt 在不同采样器 / 精度 / 框架下结果会有差异？工程上如何保证生成结果可复现？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

「同 seed + 同 prompt 但出图不同」是 SD 工程化中最常被反复追问的问题。Seed **只决定初始噪声**；从初始噪声到最终图像的链路上还有大量额外的「随机源」与「数值不一致源」。

#### 1. seed 真正决定了什么

- 初始 latent $z_T \sim \mathcal{N}(0, I)$ 的具体采样值；
- 训练 / 推理过程中所有调用 `torch.randn`、`torch.rand` 的随机数序列；
- 如果 sampler 是随机型（如 ancestral / SDE 系），每一步注入的噪声序列。

**seed 不决定**：模型权重、采样器算法、时间步离散化方式、CFG scale、CFG 形式（cond/uncond batch 顺序）、attention 实现、数值精度、GPU/CPU 后端、cudnn benchmark。

#### 2. 出现差异的常见原因

<div align="center">

| 差异源 | 说明 | 是否影响最终图 |
| --- | --- | --- |
| **采样器算法** | DDIM / DPM-Solver / Euler-A / UniPC 的更新公式不同 | 显著 |
| **采样步数** | 同采样器不同步数下的离散化误差不同 | 显著 |
| **scheduler 配置** | linear / scaled-linear / karras / lognorm shift；betas、prediction_type | 显著 |
| **精度** | fp32 / fp16 / bf16 的舍入误差累积 | 中等～显著 |
| **attention 后端** | 原生 / xFormers / SDPA / FlashAttention 的算子顺序、reduction 路径 | 轻微～中等 |
| **GPU / 驱动** | A100 / H100 / 4090 的 cuBLAS / cuDNN tile 选择不同 | 轻微 |
| **CPU 与 GPU 的 randn** | 两者实现不同，PyTorch 文档明确不保证一致 | 显著 |
| **cudnn.benchmark = True** | 会根据输入形状选最快算子，引入非确定性 | 中等 |
| **batch 内顺序与 padding** | 多 prompt 拼 batch 时不同顺序也可能改变结果 | 轻微 |

</div>

#### 3. 可复现性的工程做法

1. **冻结环境**：固定 PyTorch、CUDA、xFormers / SDPA、diffusers、模型权重哈希，最好打成镜像。
2. **统一 seed 设定**：`torch.manual_seed(seed)`、`torch.cuda.manual_seed_all(seed)`、`numpy.random.seed(seed)`、`random.seed(seed)`。
3. **关闭非确定性算子**：`torch.use_deterministic_algorithms(True)`、`torch.backends.cudnn.benchmark = False`、`torch.backends.cudnn.deterministic = True`，并按 PyTorch 文档设置 `CUBLAS_WORKSPACE_CONFIG`。
4. **统一精度**：尽量在 fp32 或同一型号 GPU 的 bf16 / fp16 下复现；跨硬件复现往往只能做到「视觉一致」，难做到 bit-exact。
5. **统一采样链路**：固定采样器、步数、scheduler 配置、CFG scale、CFG 实现（cond / uncond 是否同 batch）。
6. **A1111 / ComfyUI 复现注意点**：A1111 的「随机种子」作用于 CPU 的 `randn`，ComfyUI 默认 GPU `randn`，二者直接互换 seed 无法对齐——需要切换 `randn_source`。

**面试金句**：seed 只锁住「初始噪声」，可复现性还需要锁住「采样链路 + 数值后端 + 硬件环境」整条链。在生产环境中，复现的常见做法是：**镜像化环境 + 显式确定性配置 + 同一型号 GPU + 锁定采样器/步数/精度**，否则只能保证「视觉相似」而非「逐像素一致」。


<h2 id="q-037">面试问题：Stable Diffusion 中的 negative prompt（反向提示词）是如何加入的？</h2>

#### 1. 假想方案

容易想到的一个方案是 unet 输出 3 个噪声，分别对应无prompt，positive prompt 和 negative prompt 三种情况，那么最终的噪声就是

<div align="center"><img src="./imgs/negative_prompt_2.png" alt="negative prompt 假想方案公式" /></div>

理由也很直接，因为 negative prompt 要反方向起作用，所以加个负的系数。

#### 2. 真正实现方法

stable diffusion webui 文档中看到了 negative prompt 真正的[实现方法](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Negative-prompt)。一句话概况：将无 prompt 的情形替换为 negative prompt，公式则是

<div align="center"><img src="./imgs/negative_prompt_1.png" alt="negative prompt 实际实现公式" /></div>

就是这么简单，其实也很说得通，虽说设计上预期是无 prompt 的，但是没有人拦着你加上 prompt（反向的），公式上可以看出在正向强化positive prompt的同时也反方向强化——也就是弱化了 negative prompt。同时这个方法相对于我想的那个方法还有一个优势就是只需预测 2 个而不是 3 个噪声。可以减少时间复杂度。

<h2 id="q-038">面试问题：Stable Diffusion 中文本条件是如何一步步控制图像生成的？请完整描述从 Prompt 到 Latent 的注入链路</h2>

1.文本编码：CLIP Text Encoder模型将输入的文本Prompt进行编码，转换成Text Embeddings（文本的语义信息），由于预训练后CLIP模型输入配对的图片和标签文本，Text Encoder和Image Encoder可以输出相似的embedding向量，所以这里的Text Embeddings可以近似表示所要生成图像的image embedding。

2.CrossAttention模块：在U-net的corssAttention模块中Text Embeddings用来生成K和V，Latent Feature用来生成Q。因为需要文本信息注入到图像信息中里，所以用图片token对文本信息做 Attention实现逐步的文本特征提取和耦合。


<h2 id="q-039">面试问题：介绍 Stable Diffusion 核心网络结构</h2>

1.CLIP：CLIP模型是一个基于对比学习的多模态模型，主要包含Text Encoder和Image Encoder两个模型。在Stable Diffusion中主要使用了Text Encoder部分。CLIP Text Encoder模型将输入的文本Prompt进行编码，转换成Text Embeddings（文本的语义信息），通过的U-Net网络的CrossAttention模块嵌入Stable Diffusion中作为Condition条件，对生成图像的内容进行一定程度上的控制与引导。

2.VAE：基于Encoder-Decoder架构的生成模型。VAE的Encoder（编码器）结构能将输入图像转换为低维Latent特征，并作为U-Net的输入。VAE的Decoder（解码器）结构能将低维Latent特征重建还原成像素级图像。在Latent空间进行diffusion过程可以大大减少模型的计算量。
U-Net

3.U-net:进行Stable Diffusion模型训练时，VAE部分和CLIP部分都是冻结的，主要是训练U-net的模型参数。U-net结构能够预测噪声残差，并结合Sampling method对输入的特征进行重构，逐步将其从随机高斯噪声转化成图像的Latent Feature。训练损失函数与DDPM一致：

<div align="center"><img src="./imgs/DDPM_loss.png" alt="训练损失函数" /></div>

<h2 id="q-040">面试问题：Stable Diffusion 中的 Inpaint 和 Outpaint 分别是什么？</h2>

- **Inpaint（局部修复）** 指对图像中指定区域进行内容修复或替换的技术。用户可通过遮罩（Mask）标记需修改的区域，并输入文本提示（如“草地”或“删除物体”），模型将根据上下文生成与周围环境协调的新内容。典型应用包括移除水印、修复破损图像或替换特定对象。
- **Outpaint（边界扩展）** 则用于扩展图像边界，生成超出原图范围的合理内容。例如，将一幅风景画的左右两侧延伸，生成连贯的山脉或天空。其核心挑战在于保持扩展区域与原始图像在风格、光照和语义上的一致性。

两者均基于 Stable Diffusion 的潜在扩散模型，但目标不同：Inpaint 聚焦于“内部修正”，而 Outpaint 致力于“外部延展”，共同拓展了生成式 AI 在图像编辑中的灵活性。


<h2 id="q-040a">面试问题：Stable Diffusion 中的 img2img（图生图）原理是什么？denoising strength 起到什么作用？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

img2img 是 SD 最常用的二次创作能力，本质是 **在前向扩散链上选一个中间时刻 $t^*$ 作为起点，从这个加噪后的 latent 开始反向去噪**，而不是从纯噪声 $\mathcal{N}(0, I)$ 开始。

#### 1. 完整流程

1. **VAE 编码**：把输入参考图编码为 latent $z_0$。
2. **加噪到中间步**：根据 denoising strength $s \in [0, 1]$ 计算起始时间步 $t^* = \lfloor s \cdot T \rfloor$，然后对 $z_0$ 一步加噪：

   ```math
   z_{t^*} = \sqrt{\bar\alpha_{t^*}}\,z_0 + \sqrt{1 - \bar\alpha_{t^*}}\,\epsilon,\quad \epsilon\sim\mathcal{N}(0,I)
   ```

3. **从 $t^*$ 反向去噪**：以 $z_{t^*}$ 为起点、文本条件为引导，跑剩余的 $\lceil s \cdot \text{steps} \rceil$ 个采样步。
4. **VAE 解码**：把最终 latent 解码回像素。

#### 2. denoising strength 的作用与直觉

- $s = 0$：不加噪，模型基本「拷贝」原图。
- $s$ 较小（0.2 ~ 0.4）：保留原图大结构与构图，仅做「细节修饰、风格轻调」。常用于细节增强、轻微风格转绘、局部替换的边界融合。
- $s$ 中等（0.5 ~ 0.7）：原图作为「构图与色调骨架」，模型在此基础上做较强重绘。常用于风格迁移、人物动作迁移、参考构图二创。
- $s$ 较大（0.8 ~ 0.95）：仅保留原图的极低频信息（大体明暗、轮廓），生成结果与原图差异显著。
- $s = 1$：等价于 txt2img（从纯噪声开始）。

#### 3. 工程要点

- denoising strength 同时控制「起始 $t^*$」和「实际跑的步数」，因此 strength 越小推理越快。
- img2img 与 **Inpaint、ControlNet、IP-Adapter** 是正交能力，可以叠加使用：strength 控制原图保留度，ControlNet 控制结构，IP-Adapter 控制风格 / ID。
- 在 SDXL / SD3 上做 img2img 时，micro-conditioning（original/target size）必须传入与原图一致的尺寸，否则会出现尺寸偏差导致的细节崩溃。
- **SDEdit 论文**是 img2img 的理论起源：「在合适的中间噪声水平上加噪再去噪，可以同时保留高层语义与改变低层细节」。

**面试金句**：img2img 不是把原图「画进 prompt 里」，而是把原图当作扩散链上的一个「中间状态」，让模型从这一步继续向 $t=0$ 去噪；denoising strength 决定了「保留多少原图信息 / 模型有多少自由度」。


<h2 id="q-040b">面试问题：Stable Diffusion 训练 / 推理为什么需要 EMA（指数滑动平均）权重？常见 EMA decay 的取值与权衡是什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐ (3/5)**

EMA（Exponential Moving Average）是在训练过程中**用滑动平均的方式维护一份「平滑版」权重**：

```math
\theta_{\text{ema}}^{(t)} = \mu \cdot \theta_{\text{ema}}^{(t-1)} + (1 - \mu) \cdot \theta^{(t)}
```

最终发布与推理时使用的是 $\theta_{\text{ema}}$，而不是优化器最后一步的 $\theta$。

#### 1. 为什么扩散模型几乎必上 EMA

- **去除高频抖动**：扩散模型损失非常平坦但带高频噪声（不同 $t$ 的 loss 量级差异大），原始权重在小批量、大学习率下波动剧烈；EMA 等价于在权重空间做低通滤波，得到更接近损失「平坦谷底」的权重。
- **提升 FID / 生成质量**：在 DDPM、ADM、SDXL、SD3 论文中均有明确报告——EMA 权重相比原始权重，FID 显著下降、视觉一致性更好。
- **采样稳定性**：去噪过程对权重微小扰动敏感，EMA 减小了「同一 prompt 不同 ckpt 出图差异巨大」的问题。
- **配合 mixed precision / 大 batch**：在 fp16 / bf16 训练中，EMA 用 fp32 维护副本可以缓解低精度累积误差。

#### 2. EMA decay 的取值与权衡

<div align="center">

| decay $\mu$ | 等效平均窗口 | 适用场景 |
| --- | --- | --- |
| 0.999 | ≈1000 step | 小数据集 / 快速实验，更新快 |
| 0.9999 | ≈10000 step | 标准扩散模型训练（DDPM、ADM 默认） |
| 0.99995 ~ 0.99999 | ≈数万 ~ 十万 step | SDXL / SD3 这类大模型大数据集 |
| 自适应（Karras EMA、Power-Law EMA） | 训练初期 decay 小、后期 decay 大 | EDM2 / Karras 系列；解决「早期 EMA 滞后、后期 EMA 不够平滑」 |

</div>

#### 3. 工程注意事项

- **存储成本翻倍**：需要额外一份 fp32 EMA 权重副本；SDXL / SD3 的 EMA 单独占用约等于 base 模型大小的显存或磁盘。
- **训练初期偏置**：刚启动时 EMA 滞后，常做 **bias correction** 或在 warmup 后才开始累积 EMA。
- **EMA 与 finetune**：在已有 EMA 权重上做 LoRA / Dreambooth fine-tune 时，通常**只对 base 权重做 fine-tune，不再维护 EMA**，避免拉慢学习速度。
- **EMA vs SWA**：SWA（Stochastic Weight Averaging）是周期性等权平均；EMA 是连续指数平均。生成模型领域 EMA 更常用。

**面试金句**：EMA 不是「锦上添花」而是扩散模型的**事实标准**——它把损失景观中高频抖动滤掉，逼近平坦最优点，对 FID 与采样稳定性都有显著收益；decay 的选择与训练 step 数挂钩，大模型大数据集需要更大的 decay 与更长的等效平均窗口。


<h2 id="q-040c">面试问题：Stable Diffusion 系列在 fp32 / fp16 / bf16 / fp8 下的推理质量与显存差异是什么？工业上如何选择推理精度？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

精度选择是 SD 工程化部署的第一性问题，直接决定显存占用、吞吐、画质和支持的硬件平台。

#### 1. 四种精度的核心差异

<div align="center">

| 精度 | 位宽 | 数值范围 | 显存（相对 fp32） | 主要风险 |
| --- | --- | --- | --- | --- |
| **fp32** | 32 | $\sim 10^{\pm 38}$，约 7 位有效数字 | 100% | 几乎无；速度最慢 |
| **fp16** | 16 | $\sim 10^{\pm 5}$，约 3 位有效数字 | 50% | 上溢 / 下溢 NaN，尤其 VAE Decoder、softmax |
| **bf16** | 16 | $\sim 10^{\pm 38}$，约 2~3 位有效数字 | 50% | 范围大，几乎不上溢；精度略低 |
| **fp8（E4M3 / E5M2）** | 8 | $\sim 10^{\pm 4}$ / $10^{\pm 15}$ | 25% | 需要量化校准；对 attention / norm 敏感 |

</div>

#### 2. SD 系列在不同精度下的表现

- **fp32**：基本只用于训练 reference、调试与精度回归测试；推理几乎没人用。
- **fp16**：SD 1.x / 2.x 标配。**踩坑点**：SDXL 官方 VAE 在 fp16 下会出现 NaN / 白图，需要使用 `madebyollin/sdxl-vae-fp16-fix` 或在 VAE 部分回落到 fp32 / bf16。
- **bf16**：SD 3、FLUX、SDXL 官方推荐推理精度。bf16 范围大，几乎不会上溢；在 H100 / 4090 / RTX A 系列上有原生 Tensor Core 支持，推理吞吐比 fp16 不弱。**已成为新一代生成模型的事实默认精度**。
- **fp8**：H100 / RTX 40 系列原生支持；ComfyUI、TensorRT、NVIDIA Olive、PyTorch FP8 都已落地。SDXL / FLUX 的 fp8 量化已能在不显著降低 FID 的前提下，把显存压到约一半，推理速度提升 1.3 ~ 2 倍。但需要校准，对 attention 输出做 outlier 处理。

#### 3. 工业部署的精度选择建议

- **追求最高画质 / 调试**：bf16 全链路；VAE 在 SDXL 必须用 fp16-fix 或 bf16。
- **A100 / H100 / 4090 等大显存**：bf16 / fp8（FLUX、SDXL 已经成熟）。
- **消费级显卡（8 GB ~ 12 GB）**：fp16（注意 SDXL VAE 修复）；必要时启用 attention slicing、VAE tiling、CPU offload。
- **端侧 / 移动端**：int8 / fp8 + 蒸馏模型（SD-Lite、SDXL-Turbo、Lightning），并配合 OpenVINO / Core ML / NPU。
- **SD3 / FLUX**：bf16 是基线；需要进一步压缩时用 fp8 + 量化感知校准。

**面试金句**：扩散模型对**数值范围**比对**有效数字位数**更敏感，所以在新一代 SD3 / FLUX 上 **bf16 优于 fp16**；fp8 是当前显存与吞吐的最佳折中，但需要量化校准与 outlier 处理；fp32 仅用于训练复现与精度基准。


<h1 id="q-041">2.介绍一下 Stable Diffusion 中 VAE 的架构、原理和作用</h1>

<h2 id="q-042">面试问题：VAE 为什么会导致图像变模糊？</h2>


<h2 id="q-043">面试问题：为什么 VAE 单独做生成效果不好，但是 VAE + Diffusion 的图像生成效果就很好？</h2>

**这个问题最本质的回答是：传统深度学习时代的VAE是单独作为生成模型；而在AIGC时代，VAE只是作为特征编码器，提供特征给Diffusion用于图像的生成。其实两者的本质作用已经发生改变。**

同时传统深度学习时代的VAE的重构损失只使用了平方误差，而Stable Diffusion中的VAE使用了平方误差 + Perceptual损失 + 对抗损失。在正则项方面，传统深度学习时代的VAE使用了完整的KL散度项，而Stable Diffusion中的VAE使用了弱化的KL散度项。同时传统深度学习时代的VAE将图像压缩成单个向量，而Stable Diffusion中的VAE则将图像压缩成一个 $N \times M$ 的特征矩阵。

上述的差别都导致了传统深度学习时代的VAE生成效果不佳。


<h2 id="q-044">面试问题：Stable Diffusion 模型中的 VAE 和单纯的 VAE 生成模型的区别是什么？</h2>

#### 传统VAE生成模型

- **完整的生成系统**：从噪声直接生成数据
- **核心机制**：变分推断 + 重参数化技巧
- **目标**：学习数据分布，实现无条件生成
- **挑战**：生成质量与多样性的平衡

#### Stable Diffusiuon模型中的VAE

- **功能组件**：数据压缩器和重建器
- **核心作用**：将图像压缩到潜在空间，降低计算成本
- **目标**：高保真度重建，为扩散过程提供高效空间
- **优势**：专注重建质量，与扩散模型协同工作


<h2 id="q-044a">面试问题：从 SD 1.x → SDXL → SD 3 → FLUX.1，VAE 在通道数、下采样率、训练目标上的演进路线是怎样的？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

VAE 是连接「像素世界」与「扩散世界」的桥梁，是 SD 系列代际跃迁中持续被升级的核心组件。整体演进可以概括为三条主线：**通道数从 4 → 16、下采样率稳定在 8x、训练目标从「像素重建」走向「感知 + 对抗 + 多尺度」**。

#### 1. 主流 SD 系列 VAE 演进对比

<div align="center">

| 版本 | 输入分辨率 | 下采样率 | 通道数 $d$ | 主要损失 | 关键改进 |
| --- | --- | --- | --- | --- | --- |
| **SD 1.x VAE** | 任意 | 8x | 4 | L1 + LPIPS + KL + PatchGAN | 基线 KL-f8 VAE |
| **SD 2.x VAE** | 任意 | 8x | 4 | 同上 | 重训权重，与 SD 1.x 不通用 |
| **SDXL VAE** | 任意 | 8x | 4 | L1 + LPIPS + KL + PatchGAN，重训数据更多 | 重建细节明显提升；fp16 数值不稳，需 fp16-fix |
| **SD 3 VAE** | 任意 | 8x | **16** | L1 + LPIPS + KL + Adversarial（更新版判别器） | **通道数翻 4 倍**，显著提升小物体（人脸、文字）重建质量 |
| **FLUX.1 VAE** | 任意 | 8x | 16 | 同 SD 3 思路，配合 mean shift / scaling | 与 SD 3 类似的高通道路线；与 MM-DiT 联合优化 |

</div>

#### 2. 演进背后的三条主线

1. **通道数升级（4 → 16）**：4 通道 latent 在 64×64（对应 512×512 像素）上信息密度有限，对人眼、文字、手指、纹理等细粒度区域容易出现「重建塌缩」。SD 3 论文通过消融实验明确证明 16 通道 VAE 的 PSNR / SSIM / LPIPS 都显著优于 4 通道，这是 SD3、FLUX 高质量生成的底层支撑。
2. **下采样率稳定在 8x**：保留这个压缩率是为了在「计算量降低 64 倍」与「重建上限不至于过低」之间取平衡；继续提高（16x、32x）会让重建质量崩塌。
3. **训练目标的丰富化**：从单纯像素 L1，演化到「L1 + LPIPS + KL + 对抗」这一**多目标组合**——L1 提供像素一致性、LPIPS 提供感知质量、KL 约束 latent 分布近高斯（便于扩散）、对抗损失抑制模糊和棋盘效应。

#### 3. 为什么 SD 3 选择「通道数翻倍」而不是「下采样率减半」

- 下采样率减半（8x → 4x）会让 latent token 数量 4 倍化，所有扩散计算成本随之 4 倍化（attention 是 16 倍化），代价过大；
- 通道数翻倍只增加每个 token 的 channel 维度，扩散网络的整体计算量增长可控，且能直接提升重建上限。

**面试金句**：SD 系列 VAE 的代际演进是「**8x 下采样不动、通道数从 4 翻到 16、损失从 L1 走向感知 + 对抗**」。理解这条路线就理解了 SD 3 / FLUX 在小物体细节、文字渲染、人脸保真上跨越式提升的底层原因。


<h2 id="q-044b">面试问题：VAE 编码后为什么要乘以 scale_factor？SD 各版本的 scale_factor 是如何确定的？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

> 该问题与 [面试问题：Stable Diffusion 中的 latent scale factor（如 0.18215）有什么作用？](#q-032a) 形成「VAE 视角 vs 扩散视角」互补，本节侧重 VAE 侧的统计估计与跨版本切换实操。

#### 1. 从 VAE 输出到扩散输入的「分布对齐」

VAE Encoder 训练时只优化 $\text{Recon} + \text{KL} + \text{LPIPS} + \text{Adv}$，**并没有显式约束输出 latent 的方差恰好为 1**。当扩散模型以这个 latent 作为输入做 $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 加噪时，`scale_factor` 的作用是把 latent 的标准差缩放到接近 1，让噪声调度公式背后的「数据分布近似 $\mathcal{N}(0, I)$」假设近似成立。

#### 2. 各版本 scale_factor 的统计估计方式

```python
with torch.no_grad():
    latents = vae.encode(big_batch_of_images).latent_dist.sample()
    # 估计单标量 std
    sigma = latents.std()
scale_factor = 1.0 / sigma
# SD 3 / FLUX 的 16 通道 VAE 还需估计 mean 做 shift
shift_factor = latents.mean()
```

<div align="center">

| 版本 | scale_factor | shift_factor | 是否需要 mean shift |
| --- | --- | --- | --- |
| SD 1.x / 2.x | 0.18215 | 0 | 否 |
| SDXL | 0.13025 | 0 | 否 |
| SD 3 | 由 `vae.config.scaling_factor` 给出 | 由 `vae.config.shift_factor` 给出 | **是** |
| FLUX.1 | 由 `vae.config.scaling_factor` 给出 | 由 `vae.config.shift_factor` 给出 | **是** |

</div>

#### 3. 跨版本切换时的注意事项

- 切换 VAE 必须**同步切换 scale_factor**；不切会出现整体偏色 / 饱和度异常，严重时直接塌缩。
- SD 3 / FLUX 的 latent 公式为 `z = (raw_latent - shift_factor) * scaling_factor`，再喂给扩散网络；解码时反操作。漏掉 shift 是迁移代码时的高频踩坑点。
- 自定义 pipeline 时优先读取 `vae.config.scaling_factor` 与 `vae.config.shift_factor`，避免硬编码导致后续模型升级时 bug。
- LoRA / Dreambooth 训练时如果替换了 VAE，**训练数据预处理 / 训练 loop / 推理 pipeline 三处的 scale 都要保持一致**，否则训练目标与推理 pipeline 不匹配。

**面试金句**：scale_factor 是「VAE 实际输出方差」的倒数，目的是让 latent 分布近似 $\mathcal{N}(0, I)$，与扩散模型的噪声调度匹配；它和扩散网络是绑定的一对常量，跨版本切换 VAE 必须同步更新；SD 3 / FLUX 还引入了 shift_factor，是 16 通道 VAE 的额外 mean 校正。


<h2 id="q-044c">面试问题：SDXL VAE 在 fp16 下出现"白图 / NaN"问题的原因是什么？工业上常见的修复方案有哪些？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

SDXL 官方 VAE 在 fp16 推理下经常出现整张白图、整张黑图或 NaN，是 SDXL 工业部署最知名的踩坑点之一。

#### 1. 根因分析

- **fp16 数值范围只有约 $\pm 6.5\times 10^4$**。SDXL VAE 在解码过程中，某些中间激活值（尤其在带有 GroupNorm + 大尺寸卷积的层）会出现峰值绝对值非常大的 outlier，**超出 fp16 上限触发 inf**；inf 经过 GroupNorm / LayerNorm / sigmoid 这类算子后产生 NaN，再扩散到全图。
- 这一现象在 SD 1.x / 2.x VAE 上很少见，但 SDXL VAE 的训练数据更广、参数更新后激活分布更长尾，导致问题集中爆发。
- bf16 的范围与 fp32 同级，所以同一份 VAE 在 bf16 下几乎不会出现这个问题。

#### 2. 工业上常见的修复方案

<div align="center">

| 方案 | 思路 | 代价 |
| --- | --- | --- |
| **VAE 单独跑 fp32 / bf16** | U-Net 用 fp16，VAE 切回 fp32 / bf16 | 显存略增，速度略降；最稳妥 |
| **使用 sdxl-vae-fp16-fix** | madebyollin 重训了一份在 fp16 下数值稳定的 VAE 权重，主流社区 / diffusers 已经默认推荐 | 与官方权重等价的视觉效果，无显存代价 |
| **Force upcast** | diffusers 提供 `vae.enable_upcast()` 或 `force_upcast=True`，自动在解码时把激活上转 fp32 | 实现简单；速度略降 |
| **bf16 全链路** | H100 / 4090 / 30 系等支持 bf16 的硬件直接走 bf16 | 推荐做法，新代码默认 |
| **VAE Tiling + fp32**（极端低显存） | tiling 减少瞬时显存，VAE 仍走 fp32 | 速度损失大，仅低显存场景 |

</div>

#### 3. 工程经验

- 生产部署中**默认搭配 sdxl-vae-fp16-fix 或 bf16**，避免单点故障导致整张白图。
- 若使用 ComfyUI / A1111，绝大多数发行版都已自动选择 fp16-fix VAE 或在 VAE 层面做 upcast，不需要额外配置。
- 训练 SDXL LoRA / Dreambooth 时，VAE 推荐 fp32 或 bf16，不建议训练阶段冒险用 fp16，否则可能在数据预处理阶段就出现 NaN 样本。
- 自定义 pipeline 中要做兜底：`if torch.isnan(latents).any(): fallback_to_fp32()`。

**面试金句**：SDXL VAE 的 fp16 NaN 问题源于「fp16 数值范围太窄 + SDXL VAE 激活的长尾 outlier」；工业上的标准解法是 **bf16 全链路** 或者 **fp16 + sdxl-vae-fp16-fix**，并在 pipeline 层面加 NaN 兜底。


<h2 id="q-044d">面试问题：大分辨率推理时如何降低 VAE 解码显存？VAE Tiling 与 TAESD 各自的取舍是什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐ (3/5)**

VAE Decoder 的显存随分辨率呈 $\mathcal{O}(H \cdot W)$ 增长，是 SDXL / SD 3 / FLUX 在 1024×1024 及以上分辨率下的「显存最后一公里」。两种主流压缩方案有各自适用场景。

#### 1. VAE Tiling（分块解码）

- **思路**：把 latent 切成多个空间小块，每块独立通过 Decoder 解码，再用**重叠 + 加权融合**策略拼接成完整像素图。
- **优势**：完全无损（与一次性解码视觉一致，仅有亚像素级别的拼接差异）；不改变 VAE 权重，所有 SD 系列通用。
- **代价**：解码时间延长（每块都要单独跑一次 conv stack）；拼接边界需要羽化 / overlap，否则可能出现接缝。
- **diffusers 用法**：`pipe.vae.enable_tiling()`、可配 `tile_sample_min_size` 等参数。
- **适用场景**：1536 / 2048 / 4K 等大尺寸生成、Outpaint、超分辨率图像 latent 解码。

#### 2. TAESD / TAESDXL（Tiny AutoEncoder for SD）

- **思路**：训练一个**比官方 VAE 小一个数量级的微型 Encoder/Decoder**（通常只有几百万参数），用蒸馏方式逼近官方 VAE 的 latent 分布与重建。
- **优势**：解码极快（数倍于官方 VAE）；显存占用极小；非常适合 **Live Preview**（边采样边解码预览）和 ComfyUI 的实时小图反馈。
- **代价**：重建质量比官方 VAE 略低，**不能用于最终输出**——细节、文字、人脸的清晰度低于官方 VAE。
- **适用场景**：交互式预览、采样过程中的中间帧可视化、低端硬件的非最终输出。

#### 3. 选型建议

<div align="center">

| 场景 | 推荐 |
| --- | --- |
| 1024×1024 最终输出 | 官方 VAE + bf16 / fp16-fix |
| 2K / 4K 最终输出 | 官方 VAE + **VAE Tiling** |
| 采样过程实时预览 | **TAESD / TAESDXL** |
| 极低显存设备 | TAESD（预览） + 最终用云端官方 VAE |
| 视频生成的逐帧解码 | 官方 VAE + Tiling，并配合 fp8 / int8 量化 |

</div>

**面试金句**：VAE Tiling 用「时间换空间」做无损降显存，是大分辨率最终输出的标准方案；TAESD 用「画质换速度」做轻量解码，是实时预览与端侧的首选；二者本质是「精度优先 vs 时延优先」的不同取舍，可以叠加使用。


<h2 id="q-044e">面试问题：VAE / Tokenizer / Latent 空间为什么会影响图像生成质量和训练效率？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

VAE、Tokenizer 和 Latent 空间决定了图像从像素空间进入生成模型训练空间的方式。它们不是 Stable Diffusion pipeline 里的辅助模块，而是现代高分辨率图像生成模型的底层信息瓶颈。

Stable Diffusion、SDXL、SD 3、FLUX、Qwen-Image、Z-Image 这类模型通常不直接在像素空间训练，而是先用 VAE 或图像 Tokenizer 把图像压缩到 latent 空间，再由 U-Net / DiT / Flow 模型学习 latent 分布，最后再解码回像素图。这样做能显著降低计算成本，但也带来一个关键代价：**被 VAE 压缩丢掉的信息，后面的扩散主干很难稳定恢复。**

它对模型质量和效率的影响主要体现在四个方面：

1. **训练效率。**  
   压缩率越高，latent token 越少，U-Net / DiT 的计算量越低。对于 DiT 来说，token 数会直接影响 Attention 成本，所以 Z-Image 的紧凑 VAE、高压缩 latent 路线，本质上是在降低训练和推理成本。

2. **细节上限。**  
   如果 VAE 不能重建小字、笔画、边缘、纹理和细线结构，生成主干即使理解了 prompt，最终解码也会糊。Qwen-Image-VAE-2.0 这类面向富文本场景优化的 VAE，核心就是解决“语义知道了，但细节还原不出来”的问题。

3. **编辑保真。**  
   图像编辑要求保留原图身份、结构、背景和未编辑区域。VAE 重建质量不足时，即使编辑指令很简单，也可能出现人脸漂移、商品变形、背景纹理改变等问题。

4. **高分辨率支持。**  
   分辨率越高，latent token 越多。VAE 的压缩率、通道数、latent 尺度、scale_factor、shift_factor 和 tiling 策略，会共同决定模型能否稳定支持 2K、4K 甚至更大尺寸输出。

面试中可以这样总结：**VAE 决定“模型看见什么”和“最终能还原什么”。扩散/Flow 主干决定生成能力，VAE / Tokenizer 决定信息瓶颈；文字渲染、细节保真、编辑稳定性和推理成本，都绕不开 latent 空间设计。**


<h1 id="q-045">3.介绍一下 Stable Diffusion 中 Backbone 的架构、原理和作用</h1>

<h2 id="q-046">面试问题：Stable Diffusion 是如何在 U-Net 内部把文本与图像两种模态的语义对齐的？</h2>


<h2 id="q-047">面试问题：介绍一下 Stable Diffusion 中的交叉注意力机制</h2>

#### 1. 简介

属于Transformer常见Attention机制，用于合并两个不同的sequence embedding。两个sequence是：Query、Key/Value。

<div align="center"><img src="./imgs/cross-attention-detail-perceiver-io.png" alt="Cross-Attention 计算示意图" /></div>

Cross-Attention和Self-Attention的计算过程一致，区别在于输入的差别，通过上图可以看出，两个embedding的sequence length 和embedding_dim都不一样，故具备更好的扩展性，能够融合两个不同的维度向量，进行信息的计算交互。而Self-Attention的输入仅为一个。

#### 2. 作用

Cross-Attention可以用于将图像与文本之间的关联建立，在stable-diffusion中的Unet部分使用Cross-Attention将文本prompt和图像信息融合交互，控制U-Net把噪声矩阵的某一块与文本里的特定信息相对应。


<h2 id="q-048">面试问题：Stable Diffusion 中 cross attention 的 Q / K / V 分别是什么？为什么图像隐变量作为 Q，文本 Prompt 作为 K / V？</h2>


<h2 id="q-049">面试问题：为什么使用 U-Net 作为 Stable Diffusion 模型的核心架构？介绍一下 U-Net 架构</h2>

#### 1. U-Net的结构具有以下特点

- **整体结构**：U-Net由多个大层组成。在每个大层中，特征首先通过下采样变为更小尺寸的特征，然后通过上采样恢复到原来的尺寸，形成一个U形的结构。
- **特征通道变化**：在下采样过程中，特征图的尺寸减半，但通道数翻倍；上采样过程则相反。
- **信息保留机制**：为了防止在下采样过程中丢失信息，UNet的每个大层在下采样前的输出会被拼接到相应的大层上采样时的输入上，这类似于ResNet中的"shortcut"。

<div align="center"><img src="./imgs/unet.jpg" alt="unet" /></div>

U-Net 具有编码器部分和解码器部分，均由 ResNet 块组成。编码器将图像表示压缩为较低分辨率图像表示，并且解码器将较低分辨率图像表示解码回据称噪声较小的原始较高分辨率图像表示。更具体地说，U-Net 输出预测噪声残差，该噪声残差可用于计算预测的去噪图像表示。为了防止U-Net在下采样时丢失重要信息，通常在编码器的下采样ResNet和解码器的上采样ResNet之间添加快捷连接。

Stable Diffusion的U-Net 能够通过交叉注意力层在文本嵌入上调节其输出。交叉注意力层被添加到 U-Net 的编码器和解码器部分，通常位于 ResNet 块之间。

<div align="center"><img src="./imgs/LDMs.png" alt="Latent Diffusion Models 架构示意图" /></div>


<h2 id="q-049a">面试问题：为什么 SD U-Net 中 Self-Attention 与 Cross-Attention 主要放在中、低分辨率层？高分辨率层为何以卷积为主？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

SD U-Net 是「卷积 + 注意力」的混合架构，注意力的放置位置不是随便选的，而是**在显存 / 计算成本与语义建模能力之间的精妙折中**。

#### 1. 注意力的计算复杂度是分辨率的二次方

对于空间形状为 $H \times W$ 的特征图，自注意力的复杂度是：

```math
\mathcal{O}\bigl((HW)^2 \cdot d\bigr)
```

在 SD 1.5（latent 64×64，VAE 8x 下采样）中，U-Net 的各下采样层分辨率依次为 $64 \to 32 \to 16 \to 8$。如果在 64×64 层就放 Self-Attention，序列长度是 4096，attention 矩阵需要 $4096^2 \approx 1.6\text{M}$ 个元素；而 16×16 层的序列长度只有 256，attention 矩阵只需 $\sim 65\text{K}$ 个元素，**计算量差 256 倍**。

#### 2. 中、低分辨率更适合做语义对齐

- **高分辨率层（64×64、32×32）感受野小、语义弱**，主要承担「纹理、边缘」这类局部信息，用卷积已经足够；
- **中、低分辨率层（16×16、8×8）感受野大、语义强**，每个 token 已经聚合了较大的图像区域，正适合与文本 token 做 cross-attention 进行「语义对齐」；
- 高 / 低分辨率的注意力放置规律也符合人类视觉的「先局部纹理后整体语义」直觉。

#### 3. SD 1.x / 2.x / SDXL / SD 3 在 attention 放置上的差异

- **SD 1.x / 2.x**：U-Net 的 32×32、16×16、8×8 三个分辨率层都有 Self-Attention + Cross-Attention block，64×64 层只有卷积。
- **SDXL**：把更多的 Transformer Block 集中到中分辨率（U-Net 中部更深的 attention stack），16×16 / 8×8 层 attention 数量从 SD 1.5 的 1 个增加到多个，主要为了提升大模型容量与高分辨率细节质量。
- **SD 3 / FLUX（MM-DiT）**：彻底放弃多尺度 U-Net，改为单尺度 patchify + 全局 attention；本质上把整张图压成一个 token 序列做 Transformer，分辨率与 attention 解耦，但需要更大算力。

**面试金句**：U-Net 把 Cross-Attention 集中在中、低分辨率，是因为「语义对齐 + 二次方复杂度」两个事实必须妥协；卷积负责高分辨率局部细节，注意力负责低分辨率全局语义，这是 SD 1 / SD 2 / SDXL 共享的设计哲学。SD 3 / FLUX 通过 MM-DiT 把这条妥协推翻，但代价是显著的算力上涨。


<h2 id="q-049b">面试问题：U-Net 与 DiT / MM-DiT 在 Backbone 设计哲学上的本质差异是什么？SD 系列从 U-Net 演进到 DiT 的根本原因是什么？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

从 SDXL 到 SD 3 / FLUX 的最大跃迁就是 Backbone 从 U-Net 切换到 MM-DiT。这不只是「换模型」，而是**整个生成范式的演进**：从「卷积归纳偏置 + 局部 attention」走向「无归纳偏置 + 全局 token Transformer」。

#### 1. 设计哲学对比

<div align="center">

| 维度 | U-Net | DiT / MM-DiT |
| --- | --- | --- |
| **基本算子** | 卷积为主 + 局部 attention | 纯 Transformer，全局 attention |
| **多尺度建模** | 显式多尺度（下采样 + skip-connection） | 单一尺度（patchify 后所有 token 一起处理） |
| **归纳偏置** | 强（卷积平移等变 + 局部性） | 弱（仅有位置编码） |
| **Scaling 能力** | 有限，加深加宽收益递减 | 强，深度 / 宽度 / 数据规模都能持续提升 |
| **多模态融合** | Cross-Attention 注入文本 | Token 拼接，文本与图像在统一序列里联合 self-attention |
| **算力成本** | 中分辨率有 attention，高分辨率纯卷积，整体节省 | 全局 attention，对长序列敏感 |
| **训练数据需求** | 中等 | 大（弱归纳偏置需要更多数据） |
| **实现成熟度** | 极高（diffusers / xFormers / TensorRT 全支持） | 上升期（FlashAttention、SDPA 已成熟） |

</div>

#### 2. SD 系列从 U-Net 演进到 DiT 的根本原因

1. **Scaling Law 驱动**：Transformer 在 NLP、ViT、视频生成上反复证明「越大越好」；U-Net 在 SDXL 这一规模（≈2.6B）已经接近边际收益拐点，继续加宽 / 加深收益不显著。DiT 论文（W. Peebles, S. Xie）首次系统性地证明 Transformer 在扩散模型上同样有清晰的 Scaling Law。
2. **多模态联合建模**：MM-DiT 让文本与图像 token 在同一序列里做 self-attention，对**长 prompt、强语义、文字渲染**都更友好；U-Net 的 Cross-Attention 只能让图像 query 文本，缺乏「文本反向 query 图像」的双向信息流。
3. **统一架构、便于跨任务复用**：DiT 与视频生成（DiT for Video / Sora-类）、3D / 多模态生成（W.A.L.T、MMDiT）共用一套 Transformer 范式，更容易被复用与扩展。
4. **去除卷积的硬约束**：卷积假设平移等变性，但生成模型未必需要严格平移等变（不同分辨率、不同长宽比都要支持）；纯 Transformer + 位置编码反而更灵活。

#### 3. 代价与权衡

- DiT / MM-DiT 推理算力随分辨率快速上升，需要 FlashAttention、SDPA、序列并行等系统优化；
- 弱归纳偏置带来更高的数据需求，SD 3 / FLUX 都使用了远比 SDXL 更大的训练数据；
- 工程生态（蒸馏、ControlNet、LoRA 适配器）需要为新架构重新搭建。

**面试金句**：U-Net 强归纳偏置 + 多尺度、DiT 弱归纳偏置 + 单尺度全局 attention；演进的根本动力是**扩散模型也开始遵循 Transformer 的 Scaling Law**，加上多模态联合建模的需求，这两点共同推动 SD 系列从 SDXL 的 U-Net 走向 SD 3 / FLUX 的 MM-DiT。


<h2 id="q-049c">面试问题：Stable Diffusion 中常见的注意力加速技术（xFormers、SDPA、FlashAttention、Token Merging / ToMe）的核心思想与适用场景是什么？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

注意力是 SD 显存与时延的主要瓶颈，几种主流加速技术覆盖了「IO 优化、内存优化、token 削减」三种不同思路。

#### 1. xFormers Memory-Efficient Attention

- **核心思想**：FAIR 早期提出的 attention 内存优化，**分块计算 attention，避免显式存储完整的 $N\times N$ attention 矩阵**，把显存从 $\mathcal{O}(N^2)$ 降到 $\mathcal{O}(N)$。
- **适用场景**：A1111 / ComfyUI / diffusers 长期默认加速器；对老一代 GPU（V100、T4、3090）有显著提速；近年逐渐被 SDPA / FlashAttention-2 替代。
- **代价**：与 PyTorch 原生 attention 在数值上有微小差异，可能影响逐像素复现性。

#### 2. PyTorch SDPA（`torch.nn.functional.scaled_dot_product_attention`）

- **核心思想**：PyTorch 2.0 引入的统一接口，**根据硬件与输入自动 dispatch 到 FlashAttention、Memory-Efficient Attention 或 Math Backend**，无需安装额外库。
- **优势**：官方维护、API 稳定、长期支持；diffusers ≥ 0.20 默认启用。
- **适用场景**：所有新代码的首选；H100 / 4090 等支持 FlashAttention-2 的硬件几乎与 xFormers / FlashAttention-2 等价。

#### 3. FlashAttention / FlashAttention-2 / FlashAttention-3

- **核心思想**：**IO-aware 算法**，把 Q / K / V 切成 SRAM 友好的块，融合 softmax 与矩阵乘的访存，减少 HBM 读写次数；FA-2 改进 work partitioning，FA-3 利用 H100 的异步 TMA / WGMMA 进一步提速。
- **优势**：在新一代硬件（A100、H100、MI300）上提供数倍于原生 attention 的吞吐；与 SDPA 兼容。
- **适用场景**：训练或大批量推理时；在消费级 30 / 40 系卡上需 FA-2 + 编译。

#### 4. Token Merging（ToMe-SD）

- **核心思想**：在 self-attention 之前**把视觉相似的 token 合并为一个**，attention 序列长度变短，再在 attention 之后还原。来自 ICLR 2023 ToMe；ToMe-SD 是其在 SD 上的工程化实现。
- **优势**：不改权重、不需要重训，可叠加在 xFormers / SDPA 之上；典型设置下 SDXL 提速 ≈30-50%、轻微画质损失。
- **代价**：质量轻微下降，对细节、文字、人脸的影响比对纹理更明显；对 ControlNet / IP-Adapter 等条件控制偶尔会有副作用。
- **适用场景**：对画质要求中等的批量生成、Live Preview、移动端等。

#### 5. 其他延伸

- **DeepCache**：缓存 U-Net 中间层特征，跨多步采样复用；与 attention 加速正交。
- **SDPA + torch.compile**：PyTorch 2.x 推荐组合，能为 SDXL / FLUX 再提速 20-50%。
- **TensorRT / ONNX Runtime**：把 attention 融合进图编译，工业部署常用。

**面试金句**：xFormers / SDPA / FlashAttention 三者本质都是**「同语义、不同实现」的 IO 优化**，属于「不损画质换显存与速度」；Token Merging 是**「主动丢信息」的真减法**，属于「轻微损画质换大幅提速」。生产部署的最佳实践是 **SDPA（FlashAttention 后端） + ToMe（可选） + torch.compile**。


<h2 id="q-049d">面试问题：SD Backbone 中 GroupNorm + SiLU + 残差连接的设计为何对训练稳定性很关键？换成 LayerNorm / BatchNorm 会有什么问题？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐ (3/5)**

SD U-Net 中的每一个 ResBlock 都是「GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → 残差加法 + timestep / 文本调制」。这套组合不是任意选择，而是扩散模型训练稳定性的「最小刚需配方」。

#### 1. 为什么是 GroupNorm 而不是 BatchNorm

- **BatchNorm 依赖 batch 内统计**，扩散模型训练时同一 batch 内不同样本对应**不同的时间步 $t$**，统计分布差异巨大；BN 会把高噪样本与低噪样本混在一起做归一化，引入严重的统计偏置。
- BN 在小 batch / 推理 batch=1 时表现差；扩散模型推理常常 batch=1（CFG 时 batch=2），BN 不友好。
- **GroupNorm 只在通道维做分组归一化，与 batch 无关**，对任意 batch size 表现一致；与 timestep / 噪声水平也解耦。
- 工业中也有人使用 **AdaGN / AdaLN-Zero**（让 timestep 调制 GroupNorm 的 scale / bias），是 ADM、DiT、SD 3 的常见做法。

#### 2. 为什么是 SiLU 而不是 ReLU

- ReLU 在负区间梯度恒为 0，深层网络训练易出现「dead ReLU」；
- **SiLU（Swish）= $x \cdot \sigma(x)$**：在负区间有非零梯度，平滑可导，与扩散模型「连续噪声水平」的特性更匹配；
- DDPM 原论文消融显示 SiLU 比 ReLU、GELU 都更稳定；ADM / SD 系列沿用至今。

#### 3. 残差连接的双重价值

- **梯度传播**：扩散网络往往很深（SDXL U-Net 有数十个 ResBlock），残差连接保证梯度能直通到深层；
- **保留高频信号**：去噪过程要求网络能处理「输入 ≈ 输出」的极端情况（高 SNR 时几乎是恒等映射），残差连接提供了这种「恒等近似」的捷径。

#### 4. 为什么不是 LayerNorm

- LayerNorm 对每个空间位置独立做归一化，破坏空间相关性，对卷积视觉任务不友好；
- 但在 **DiT / MM-DiT** 中由于 Backbone 已经是 Transformer 范式（每个位置就是一个 token），LayerNorm（含 AdaLN-Zero）反而是默认选择；这进一步说明「归一化层 ↔ Backbone 范式」存在强绑定关系。

**面试金句**：扩散模型的训练稳定性高度依赖「**与 batch 解耦的归一化（GN）+ 平滑激活（SiLU）+ 残差通路**」三件套；BatchNorm 与 timestep 多噪声共存矛盾，LayerNorm 适合 Transformer 范式但不适合卷积 U-Net；这套配方是从 DDPM 一路传承到 SDXL 的事实标准。


<h1 id="q-050">4.介绍一下 Stable Diffusion 中 Text Encoder 的架构、原理和作用</h1>

<h2 id="q-051">面试问题：举例介绍一下 Stable Diffusion 模型进行文本编码的全过程</h2>


<h2 id="q-052">面试问题：Stable Diffusion 如何通过文本来实现对图像生成内容的控制？SD 中是如何注入文本信息的？</h2>


<h2 id="q-053">面试问题：Negative Prompt 实现的原理是什么？</h2>


<h2 id="q-053a">面试问题：CLIP Text Encoder 的 77 tokens 长度限制对长 Prompt 的实际影响是什么？工程上如何突破（chunking、weighted prompt、T5 等长上下文编码器）？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

CLIP Text Encoder 的位置编码上限是 77 tokens（包含起止符 `<|startoftext|>`、`<|endoftext|>`），这是 SD 1.x / 2.x / SDXL 的硬约束，也是「用户写了一段长 prompt 但模型只听了前面几句」的根因。

#### 1. 77 tokens 限制的实际影响

- **直接截断**：超过 77 token 的 prompt 默认被截断，后半段完全丢失，模型生成的图像与用户预期不符；
- **CLIP token ≠ 单词数**：CLIP 的 BPE 分词使一个英文单词常占 1~2 token，中文单字常占 2~3 token，**77 token 实际只够 30~50 个英文单词或 25~30 个中文字**；
- **写复杂场景受限**：多角色、多物体、多风格描述无法在 77 token 内充分表达；
- **对 SD 3 / FLUX 仍然部分受影响**：SD 3 同时使用了 CLIP-L、CLIP-G（77 token）+ T5-XXL（最大 512 token），但 CLIP 部分仍受 77 token 限制，T5 才是长上下文的承载者。

#### 2. 三类主流的工程突破方案

##### (1) Prompt Chunking（A1111 / ComfyUI 通用）

- 把长 prompt 按 75 个有效 token 分块（每块再加起止符变 77）；
- 每块独立过 CLIP Text Encoder 得到 77×768 / 77×1280 的 embedding；
- 在 token 维上拼接（concat）成 $(77 \cdot N) \times d$；
- U-Net 的 cross-attention 直接接受任意长度的文本序列（attention 对序列长度本来就没限制）。
- 优势：实现简单、无需训练；劣势：跨块语义关联较弱，且 CLIP 没在长序列上训练过，超过几块后效果会衰退。

##### (2) Prompt Weighting（权重语法）

- 通过 `(word:1.3)` / `[word]` 等语法对特定 token 的 embedding 做缩放或插值；
- 不增加 token 数，而是在有限长度下「加重重要部分」。
- 见 [面试问题：Prompt 中的权重语法](#q-053b) 详细解析。

##### (3) 切换到 T5 / 长上下文 LLM 编码器

- **T5-XXL**：DALL-E 3、Imagen、SD 3、FLUX.1 都引入了 T5-XXL 作为辅助 Text Encoder，最长支持 512 token，能容纳长 prompt 的细节；
- **PixArt-α / PixArt-Σ**：直接只用 T5-XXL，单一编码器处理长 prompt；
- **未来趋势**：FLUX.2、Stable Diffusion 3.5 Large 等已实验性引入更长上下文 LLM 作为编码器。

#### 3. 工程经验

- A1111 / ComfyUI 默认开启 chunking，普通用户感受不到 77 限制；
- 对极长 prompt，SDXL 的实际「有效信息上限」其实在 100~150 token 左右；超过部分的细节被稀释；
- 当语义关联非常重要（多角色、多场景）时，更好的做法是**先用 LLM 改写 prompt**，把语义最关键的内容压缩到前 75 token，而不是无限堆 token；
- SD 3 / FLUX 上的 prompt 长度优势主要来自 T5-XXL，不要把长 prompt 同样拿去 SDXL 跑。

**面试金句**：77 token 是 CLIP 的位置编码硬限制，SD 1 / 2 / XL 通过 chunking + 权重语法做到「能跑长 prompt」但**信息密度严重稀释**；SD 3 / FLUX 通过引入 T5-XXL 才真正打破了「短 prompt」时代。理解这一点能在面试中清晰回答「为什么 SD 3 文本一致性比 SDXL 强」。


<h2 id="q-053b">面试问题：Prompt 中的权重语法（(word:1.2)、[word]）的实现原理是什么？A1111 / ComfyUI / Compel 三种 Prompt 解析方式有何差异？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

「权重语法」是社区在 SD 1.x 时代发明的事实标准，用于在不增加 token 数的前提下放大某段文本的影响力。它的实现并不在模型权重里，而在 **「prompt → embedding」这一步的 embedding 加工层**。

#### 1. 主流权重语法

<div align="center">

| 语法 | 含义 | 默认权重 |
| --- | --- | --- |
| `(word)` | 提升权重 | ×1.1 |
| `((word))` | 嵌套提升 | ×1.21 |
| `[word]` | 降低权重 | ×0.91 |
| `(word:1.5)` | 显式权重 | ×1.5 |
| `[word:1.5]` | 显式降低 | ÷1.5 |
| `(prompt_a:0.6 AND prompt_b:0.4)` | 多 prompt 加权混合（A1111 AND） | 自定义 |

</div>

#### 2. 实现原理：在 token embedding 上做缩放或插值

A1111 / ComfyUI 的核心做法都遵循以下三步：

1. **解析权重**：把 `(word:1.5)` 解析为 `(token_ids, weight)` 元组列表；
2. **取出该段 token 的 embedding**：用 CLIP 的 token embedding 表得到原始 embedding；
3. **加权处理**：常用方法有两种：
   - **均值偏移法（A1111 默认）**：`emb = mean + weight * (emb - mean)`，保持整个 prompt 的全局均值不变，避免单段权重过大导致全局偏色；
   - **直接缩放法（早期实现）**：`emb = emb * weight`，简单但容易让某段过强而压制其它部分。

#### 3. 三种实现的差异

<div align="center">

| 实现 | 解析风格 | 权重处理 | 多 prompt 混合 | 典型坑点 |
| --- | --- | --- | --- | --- |
| **A1111 / WebUI** | 字符级解析、`( ) [ ] : AND` 全支持 | 均值偏移法 | `AND` 关键字（compositional） | `(:0)` 等极端权重可能 NaN |
| **ComfyUI（原生）** | 节点式 + 文本权重；`(word:weight)` | 直接缩放法（更接近 Compel） | 通过 `ConditioningCombine` 等节点显式做 | 与 A1111 的同 prompt 出图存在差异 |
| **Compel（diffusers 生态）** | 解析更严格、支持 emphasis tree、prompt blending | 多种权重模式可配 | `prompt1.and(prompt2, weights=[...])` | 与 A1111 行为不完全等价 |

</div>

#### 4. 工程经验

- **A1111 与 ComfyUI 的 prompt 不能直接互换**：同一段 `(masterpiece:1.3)` 在两边的视觉效果会有差异，迁移工作流时需要重新调权重；
- **极端权重危险**：`(word:0)` 可能让该 token embedding 远离 mean，干扰其它 token 的归一化；建议权重区间 `[0.5, 1.5]`；
- **Negative prompt 的权重也用同一套语法**，规则一致；
- **SD 3 / FLUX 上，权重语法仍然在 CLIP 编码部分有效，但 T5 部分通常不响应**——T5 没有内置 emphasis 概念，依靠语言本身（如「very bright」）表达强度更稳定。

**面试金句**：权重语法是「在 prompt → embedding 阶段对 token embedding 做缩放或与均值插值」的工程技巧；A1111 / ComfyUI / Compel 在解析与缩放策略上的差异，导致同 prompt 跨实现不可逐像素复现，但思路一致。


<h2 id="q-053c">面试问题：CLIP skip 是什么？为什么社区在 SD 1.5 上常用 clip_skip=2，但 SDXL / SD 3 不再推荐？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐ (3/5)**

CLIP skip 是 SD 社区的一个流传度极高、但很多人「只会用不知原理」的参数。它指的是**在使用 CLIP Text Encoder 时跳过最后 N 层 Transformer，使用倒数第 N+1 层的隐藏态作为文本条件**。

#### 1. CLIP Text Encoder 的层级结构

CLIP Text Encoder 是一个 12 层（ViT-L）或 23 层（ViT-H、ViT-bigG）的 Transformer：

- **越靠近最后一层**，特征越偏「**判别性 / 对齐对比学习目标的语义特征**」（CLIP 训练目标是文本-图像对齐）；
- **倒数第二层**保留更多「**语言细节、近词关系、语法结构**」，对生成模型更友好。

#### 2. clip_skip 的具体含义

- `clip_skip=1`：使用最后一层输出（默认行为）；
- `clip_skip=2`：使用倒数第二层输出，跳过最后一层；
- `clip_skip=N`：使用倒数第 N 层。

社区上 SD 1.5 二次元 / NovelAI 系模型（如 Anything-V3、AbyssOrangeMix）默认推荐 `clip_skip=2`，因为：

- NovelAI 早期发布时**就是用 `clip_skip=2` 训练的**，匹配训练时的特征层即可获得最佳效果；
- 倒数第二层的特征更细，对动漫人物的服饰、姿势细节描述更敏感；
- 在 prompt 中包含大量描述词时，倒数第二层对每个词的响应更均衡。

#### 3. 为什么 SDXL / SD 3 不再推荐 clip_skip

- **SDXL 的训练**：SDXL 训练时**默认使用最后一层**（CLIP-L 与 OpenCLIP-bigG 的最后一层 + Pooled embedding），架构与训练目标都已围绕这一选择优化；强行 clip_skip=2 会让推理特征与训练分布错位，反而画质下降；
- **SD 3 的训练**：SD 3 用 CLIP-L、OpenCLIP-bigG 的**倒数第二层 hidden state** 作为细粒度特征 + 各自的 Pooled embedding 作为全局特征 + T5-XXL 最后一层的 hidden state，整套设计已经把「层选择」固定了，用户无需也不应该额外做 clip_skip；
- **SDXL / SD 3 的训练数据更多更高质量**：CLIP 最后一层在足够大的训练样本下也能学到细节，clip_skip 的「画质收益」就没了。

#### 4. 工程经验

- 用 SD 1.5 二次元模型时，先看模型卡片是否标注「clip_skip=2」；A1111 / ComfyUI 默认值常为 1，不切换会有明显画风偏差；
- 用 SDXL / SD 3 系列时**保持默认**（clip_skip=1 / N/A），不要听信「调 clip_skip 提升画质」的旧经验；
- 训练 LoRA / Dreambooth 时**训练用什么 clip_skip，推理就要用什么**，否则训练目标与推理 pipeline 不匹配。

**面试金句**：clip_skip 的本质是「选用 CLIP 第几层的隐藏态作为文本条件」，**层选择必须与训练时一致**；SD 1.5 二次元社区因 NovelAI 历史原因常用 clip_skip=2，但 SDXL / SD 3 由官方训练范式决定了固定的层选择，clip_skip 不再是用户应该调的参数。


<h2 id="q-053d">面试问题：为什么 SD 1.x 选用 CLIP ViT-L 而 SD 2.x 切换为 OpenCLIP ViT-H？这一切换给生成效果带来了哪些可观察的差异？</h2>

**难度评分：⭐⭐⭐ (3/5)  |  考察频率：⭐⭐⭐ (3/5)**

SD 1.x 与 SD 2.x 在生成效果上「人物画风差异巨大」，背后最直接的原因不是 U-Net 改了多少，而是 **Text Encoder 从 CLIP ViT-L/14 切换到了 OpenCLIP ViT-H/14**。这一切换是 SD 系列历史上最有争议、也最有教育意义的一次代际选择。

#### 1. 两者的核心差异

<div align="center">

| 维度 | CLIP ViT-L/14（OpenAI） | OpenCLIP ViT-H/14（LAION） |
| --- | --- | --- |
| 训练数据 | OpenAI WIT（4 亿对，闭源） | LAION-2B（20 亿对，开源） |
| 文本嵌入维度 | 768 | 1024 |
| 数据质量 | 经过严格过滤、风格分布偏精修 | 大规模网络抓取、风格分布更广但更杂 |
| 内容覆盖 | NSFW / 名人 / 艺术家被刻意过滤 | 也做了过滤，但相对宽松 |
| 文本理解 | 中等 | 略强 |

</div>

#### 2. SD 2.x 切换 OpenCLIP 的根本原因

- **可商用 / 开源合规**：OpenAI 的 CLIP 权重虽公开，但许可与训练数据不完全开放；OpenCLIP 在 LAION 上完全可重训、可商用；
- **可重训 / 可复现**：Stability 希望整个 pipeline 都是「可由社区独立训练」的，OpenCLIP 与 LAION 数据天然兼容；
- **更大模型容量**：ViT-H/14 比 ViT-L/14 更深更宽，理论上文本理解更强；
- **更好的多语言潜力**：OpenCLIP 的多语言版本（XLM-CLIP）也是 Stability 后续布局的一部分。

#### 3. 切换后的可观察差异

- **画风偏差大**：很多 SD 1.5 时代沉淀的 prompt 在 SD 2.x 上效果完全不同，甚至「画不出某些 artist 风格」（数据过滤所致）；
- **NSFW / 人物 / 名人能力下降**：训练数据过滤更严，SD 2.x 在人物面部、名人脸的生成能力相比 1.5 有所下降，是社区诟病的主因；
- **构图与色调改变**：因为 cross-attention 接收到的语义信号分布改变了，模型对同一 prompt 的语义响应也变了；
- **生态断层**：SD 1.5 上的 LoRA / 模型与 SD 2.x 不通用，导致 SD 2.x 时代社区生态远不如 SD 1.5 繁荣，**这一现象直接影响了 SDXL 的设计——SDXL 没有放弃 CLIP-L，而是 CLIP-L + OpenCLIP bigG「双 Text Encoder」并存以兼容旧 prompt**。

#### 4. 经验教训

- **Text Encoder 是 SD 系列的「DNA 层」**：换 Text Encoder 不只是换模型，而是换掉了模型对自然语言的理解空间；
- **数据过滤策略**比模型容量对生成内容覆盖范围影响更大；
- SDXL 的「双 CLIP」、SD 3 的「三编码器」都是这一教训的延续——通过组合不同 Text Encoder，**既保留旧生态、又获得新能力**。

**面试金句**：SD 1 → SD 2 切换 OpenCLIP 是出于**开源合规 + 可重训 + 更大容量**的考虑，但带来了「画风断层 + 生态断层」的副作用；SDXL 的双 Text Encoder、SD 3 的三 Text Encoder 都是这一历史经验的工程化反思。


<h2 id="q-053e">面试问题：Text Encoder 和 VLM 条件编码器在图像生成模型中起什么作用？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

Text Encoder 或 VLM 条件编码器决定模型如何理解用户输入。早期 Stable Diffusion 主要依赖 CLIP 文本编码器，把 prompt 编码成可供 U-Net cross-attention 使用的文本特征；Imagen、DALL-E 3、SD 3、FLUX 则说明，更强的 T5 / LLM 级文本编码器会显著提升复杂 prompt following；Qwen-Image、Qwen-Image-Edit 进一步把 VLM 作为语义入口，用于理解图像、文字、布局和编辑意图。

它们的作用可以拆成五层：

1. **把 prompt 转成语义条件。**  
   文本编码器负责把自然语言 prompt 映射成条件向量或 token 序列，再通过 cross-attention、AdaLN、joint attention、MMDiT 等机制注入生成主干。没有稳定的文本条件，扩散模型只能学无条件图像分布。

2. **理解长 prompt 和复杂关系。**  
   CLIP 擅长短文本图文对齐，但对长描述、多对象关系、空间逻辑、段落级排版和复杂否定约束较弱。T5、LLM 和 VLM 可以补足这部分能力，这也是 SD 3、FLUX、Qwen-Image 这类模型强化文本/多模态编码器的重要原因。

3. **支持图像编辑和多模态输入。**  
   图像编辑不仅要理解“把什么改成什么”，还要理解源图里已有的主体、身份、结构和布局。Qwen-Image-Edit 这类路线通常会同时利用 VAE Encoder 保留外观结构、利用 VLM 条件编码器理解图像语义，从而让模型既知道“要改什么”，也知道“原图长什么样”。

4. **支持文字渲染、排版和知识图像。**  
   海报、菜单、PPT、流程图、信息图、地图和公式图不只是“画出类似图案”，还要求文字内容、布局层级和逻辑关系正确。强文本编码器/VLM 能把文字、OCR、版式和对象关系以更高密度送入生成主干，是富文本图像生成能力提升的关键。

5. **影响生态兼容和模型代际差异。**  
   SD 1.x、SD 2.x、SDXL、SD 3、FLUX 在 prompt 行为上的差异，很大一部分来自 Text Encoder 的差异。换编码器不是简单换模块，而是换掉模型对自然语言的理解空间，也会影响 LoRA、prompt 模板、clip_skip、长 prompt 处理和社区工作流兼容性。

一句话总结：**现代图像生成模型越来越像“语言/多模态理解模型 + 视觉生成模型”的组合。Text Encoder / VLM 决定模型听不听得懂，U-Net / DiT / Flow 主干决定模型画不画得出来。**


<h2 id="q-054">面试问题：如何处理 Prompt 和生成的图像不对齐的问题？</h2>


<h2 id="q-055">面试问题：扩散模型是如何引入控制条件的？</h2>

在现代扩散模型中，引入控制条件的方式主要分为两大类：**采样阶段的引导（Guidance）与网络结构级的条件融合（Architectural Conditioning）**。前者通过调整去噪过程中的梯度方向，在不改动模型参数的前提下实现条件控制；后者则在模型内部直接注入额外信息，包括跨注意力（Cross‐Attention）和时间嵌入（Time Embedding）的多路拼接。下面我们将从这两大类出发，详细介绍包括交叉注意力注入、时间步嵌入拼接、类别嵌入拼接以及 ControlNet 等多种常见的条件引入技术。

#### 一、采样阶段的引导方法

##### 1.1 分类器引导（Classifier Guidance）

- **原理**：额外训练一个图像分类器，对去噪过程中的中间图像计算类别概率梯度 $\nabla\log p(y\mid x)$ ，并将其与扩散模型的去噪梯度相加，以朝着目标类别 $y$ 的方向更强地去噪。
- **特点**：无需改变原扩散模型结构，可后期直接应用；但需额外训练分类器，且计算开销较大。

##### 1.2 无分类器引导（Classifier-Free Guidance）

- **原理**：在同一模型中联合训练"有条件"（带 $y$ 输入）与"无条件"（不带 $y$ ）的分支，采样时按比例 $s$ 调整两者的去噪预测：

```math
\hat{\epsilon}=(1+s)\epsilon_{\mathrm{cond}}-s\,\epsilon_{\mathrm{uncond}}
```

通过增大 $s$ ，可在样本质量与多样性间权衡。

- **优势**：无需单独训练分类器，已成为文本到图像任务的主流引导策略。

#### 二、网络结构级的条件融合

##### 2.1 跨注意力（Cross-Attention）注入

- **文本到图像**：在每个 U-Net 模块的中间，使用跨注意力层将文本嵌入（如 CLIP 编码）作为键/值，图像特征作为查询，实现与自然语言条件的交互。
- **多模态扩展**：可将其它概念 token（如布局、分割图等）也作为条件序列，通过相同机制注入，支持更灵活的条件输入。

##### 2.2 时间步嵌入（Time Embedding）拼接

- **位置编码**：采用类似 Transformer 的正余弦编码映射时间步 $t$ 到向量 $\text{pos}(t)$ ，然后通过线性层得到时间嵌入。
- **融合方式**：除常见的**加法融合**外，也可将时间嵌入与其它条件（如类别 embedding 或空间特征）在通道维度上**拼接**，再一起输入至卷积层或注意力模块中。

##### 2.3 类别嵌入（Class Embedding）拼接

- **方法**：将类别 embedding（CEN）在每层噪声估计器（noise estimator）中与特征张量**串联**（concatenate），使得扩散的重建过程同时感知图像内容与类别信息。
- **效果**：在多类别生成任务中，可显著提升类别一致性，同时保持图像质量。

##### 2.4 ControlNet：条件分支并行注入

- **原理**：在预训练 U-Net 的每个编码器层复制一份"可训练"分支，并通过零初始化卷积（ZeroConv）接收额外条件（如边缘图、深度图），其输出再**加回**主干层，确保不破坏原模型能力。
- **应用**：广泛用于 Stable Diffusion，为图像生成提供细粒度空间控制，如姿态、分割或布局指令。

#### 三、其他控制技术

- **Cross-Attention Score 调整**：在生成时对跨注意力分数进行训练无关的修改，以强化局部概念在图像中的表现，同时避免语义混合（concept bleeding）。
- **CFG++等高级引导**：在无分类器引导基础上优化 off-manifold 轨迹，提升高引导尺度下的可逆性与样本质量。


<h1 id="q-056">5.Stable Diffusion XL 有哪些创新点？</h1>

<h2 id="q-057">面试问题：与 Stable Diffusion 相比，Stable Diffusion XL 的核心优化有哪些？</h2>

1、模型参数更大。SDXL 基础模型所使用的 Unet 包含了2.6B（26亿）的参数，对比 SD1.5的 860M（8600万），相差超过三倍。因此从模型参数来看，SDXL 相比 SD 有显著优势。

2、语义理解能力更强。使用了两个 CLIP 模型的组合，包括 OpenClip 最大的模型 ViT-G/14 和在 SD v1 中使用的 CLIP ViT-L，既保证了对旧提示词的兼容，也提高了 SDXL 对语言的理解能力

3、训练数据库更大。由于 SDXL 将图片尺寸也作为指导参数，因此可以使用更低分辨率的图片作为训练数据，比如小于256x256分辨率的图片。如果没有这项改进，数据库中高达39%的图片都不能用来训练 SDXL，原因是其分辨率过低。但通过改进训练方法，将图片尺寸也作为训练参数，大大扩展了训练 SDXL 的图片数量，这样训练出来的模型具有更强的性能表现。

4、生图流程改进。SDXL 采用的是两阶段生图，第一阶段使用 base model（基础模型）生成，第二阶段则使用 refiner model（细化模型）进一步提升画面的细节表现。当然只使用 SDXL 基础模型进行绘图也是可以的。

<h2 id="q-058">面试问题：Stable Diffusion XL 的 VAE 部分有哪些创新？详细分析改进意图</h2>


<h2 id="q-059">面试问题：Stable Diffusion XL 的 Backbone 部分有哪些创新？详细分析改进意图</h2>


<h2 id="q-060">面试问题：Stable Diffusion XL 的 Text Encoder 部分有哪些创新？详细分析改进意图</h2>


<h2 id="q-061">面试问题：Stable Diffusion XL 中使用的训练方法有哪些创新点？</h2>


<h2 id="q-061a">面试问题：SDXL 的 micro-conditioning（original_size / crop_top_left / target_size）是什么？为什么是 SDXL 工程化层面最关键的创新之一？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

micro-conditioning 是 SDXL 论文中最有工程价值的「不起眼小创新」，它把图像在数据预处理阶段被「裁剪、缩放、填充」过的事实**显式告诉模型**，从而修复了 SD 1.5 时代普遍存在的「人物头部缺失、构图不全、低分辨率训练数据被错误利用」等问题。

#### 1. 三个条件的具体含义

<div align="center">

| 条件 | 含义 | 训练时的取值 | 推理时建议 |
| --- | --- | --- | --- |
| `original_size` | 图像在 resize 到训练分辨率前的**原始分辨率** | 数据集真实原始尺寸 | 想让模型按高分辨率风格生成时设大值（如 1024×1024 或更大） |
| `crop_top_left` | 训练时为对齐分辨率所做的中心裁剪偏移量 $(c_y, c_x)$ | 真实裁剪偏移 | 通常设 (0, 0) 表示「无裁剪、构图完整」 |
| `target_size` | 模型最终要生成的目标分辨率 | 训练分辨率 | 与 `original_size` 一致或按需求设定 |

</div>

注入方式：将三个条件分别经过 sinusoidal 时间嵌入 → MLP，**与 timestep embedding 相加**作为全局调制信号（类似 ADM 的 timestep injection）。

#### 2. 为什么是 SDXL 工程化层面最关键的创新

1. **修复「数据浪费」问题**：SD 1.x 训练时小于训练分辨率的样本会被丢弃，损失约 39% 的训练数据；SDXL 把 `original_size` 作为条件，让模型「知道这是低分辨率原图」，所有数据都能用，扩大了训练集规模与多样性；
2. **修复「裁剪导致内容缺失」问题**：SD 1.x 中心裁剪会让人物头部 / 物体边缘被切掉，模型学到「人物没有头是合理的」；SDXL 通过 `crop_top_left` 显式标注裁剪偏移，模型在推理时设 (0, 0) 就能生成完整构图；
3. **解锁「分辨率风格控制」**：推理时把 `original_size` 设为高值（如 4096×4096）能让模型按「高清原图风格」生成；设低值则生成颗粒感更强的「低分辨率风格」；
4. **零额外训练成本**：仅在 conditioning 部分加几个 MLP，参数量增量可忽略，但收益巨大。

#### 3. 工程经验

- 推理时**忘记设 `crop_top_left=(0,0)`** 是常见踩坑：会使生成图带有训练时随机裁剪的「构图偏移」；
- diffusers 默认 pipeline 已经做好这些参数管理，但自定义 pipeline / 训练脚本时务必显式传入；
- LoRA / Dreambooth 在 SDXL 上微调时，micro-conditioning 必须保持与训练时一致；
- SDXL Refiner 也复用了 micro-conditioning，但额外加了 `aesthetic_score` 条件。

**面试金句**：micro-conditioning 是 SDXL 把「数据预处理事实」从隐性变成显性，让模型「知道训练样本被怎么处理过」；这一步既扩大了可用训练数据 39%，又解决了人物头部缺失等典型构图问题，是 SDXL 工程化最高 ROI 的小改进。


<h2 id="q-061b">面试问题：SDXL 的双 Text Encoder + Pooled Embedding 注入机制具体是怎样的？工程实现中有哪些容易忽略的细节？</h2>

**难度评分：⭐⭐⭐⭐ (4/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

SDXL 把 SD 1.x 的「单一 CLIP-L」升级为「**CLIP-L + OpenCLIP-bigG 双编码器 + Pooled Embedding 全局注入**」，这套设计在 SD 3、FLUX 中也得到延续，是新一代生成模型「文本注入」的事实模板。

#### 1. 完整注入链路

```
输入 Prompt
   │
   ├──► CLIP-L Text Encoder ─► 倒数第二层 hidden state，shape = (77, 768)
   │                            └─► Pooled Embedding（[EOT] token），shape = (768,)
   │
   └──► OpenCLIP-bigG Text Encoder ─► 倒数第二层 hidden state，shape = (77, 1280)
                                      └─► Pooled Embedding，shape = (1280,)

细粒度路径：
   concat([CLIP-L 序列, OpenCLIP-bigG 序列], dim=-1) → shape = (77, 2048)
   → 作为 U-Net Cross-Attention 的 K / V

全局路径：
   concat([CLIP-L Pooled, OpenCLIP-bigG Pooled], dim=-1) → shape = (2048,)
   → 经 MLP → 与 timestep embedding 相加 → 作为全局调制信号
```

#### 2. 双编码器的设计意图

- **CLIP-L 兼容旧生态**：保留 SD 1.x 时代用户熟悉的 prompt 行为；
- **OpenCLIP-bigG 提供更强语义**：参数量 695M，远大于 ViT-L 的 124M，对长 prompt、复杂语义、多概念组合的理解更强；
- **Pooled Embedding 提供全局风格**：cross-attention 偏向局部对齐，pooled embedding 提供整体风格 / 主题约束，二者互补；
- **细粒度 + 全局**双通路是 SD 3、FLUX 后续延续的设计模板。

#### 3. 工程实现中的常见踩坑

1. **取「倒数第二层」而非最后一层**：SDXL 训练时用的是两个编码器的**倒数第二层**hidden state；如果手写 pipeline 误用最后一层，画质会显著下降。diffusers 的 `output_hidden_states=True` 是必备配置；
2. **Pooled Embedding 不是 CLS token**：CLIP 的 pooled embedding 是**[EOT] (End of Text) token 位置**的 hidden state，**不是** [CLS]，写代码时容易搞错；
3. **两个编码器的 padding 策略要一致**：均要 padding 到 77 tokens，且 attention_mask 一致；
4. **micro-conditioning 必须和 pooled embedding 一起经过 MLP 后再叠加 timestep embedding**：顺序错了画质会塌；
5. **Prompt 在两个编码器中是同一份文本**：不是分别输入「主体 prompt」和「风格 prompt」；
6. **CFG 时双编码器的 uncond 也要分别准备**：把两个编码器分别跑空 prompt 得到对应的 uncond embedding，再 concat；
7. **训练 LoRA / Dreambooth 时，两个编码器是否一起 fine-tune** 是工程选择：通常只 fine-tune U-Net，两个 Text Encoder 冻结。

**面试金句**：SDXL 的文本注入是「**两个 CLIP 取倒数第二层 + concat 通道做细粒度 cross-attention，两个 Pooled Embedding + concat 后 + micro-conditioning 一起做全局调制**」，**取层、取 token、padding、CFG uncond 准备**这四个细节是手写 pipeline 时最容易踩坑的地方。


<h2 id="q-062">面试问题：训练 Stable Diffusion XL 时为什么要使用 offset Noise？</h2>


<h2 id="q-063">面试问题：介绍一下 Stable Diffusion XL Turbo 的原理</h2>


<h2 id="q-064">面试问题：SDXL-Turbo 用的蒸馏方法是什么？</h2>

论文链接：[adversarial_diffusion_distillation.pdf](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf)

#### 方法结构

ADD 模型的结构包括三个核心组件：

1. **ADD 学生模型 (Student Model)**：这是一个预训练的扩散模型，负责生成图像样本。
2. **判别器 (Discriminator)**：用来区分生成的样本和真实图像，通过对抗性训练来提升生成图像的真实感。
3. **DM 教师模型 (Teacher Model)**：这是一个冻结权重的扩散模型，作为知识的教师，为学生模型提供目标图像来实现知识蒸馏。

<div align="center"><img src="./imgs/SD-Turbo.jpg" alt="SD-Turbo ADD 蒸馏方法结构示意图" /></div>

#### 核心原理

ADD 的核心原理是通过两个损失函数的结合实现蒸馏过程：

1. **对抗性损失 (Adversarial Loss)**：学生模型生成的样本被输入判别器，判别器尝试将生成的样本与真实图像区分开。学生模型则优化生成图像，使其更难被判别器检测到为假，从而提升图像的细节和逼真度。
2. **蒸馏损失 (Distillation Loss)**：ADD 使用另一个扩散模型作为教师模型，并通过蒸馏损失指导学生模型生成与教师模型相似的图像。教师模型对学生生成的噪声数据进行去噪，从而提供高质量的生成目标。这样，学生模型能够利用教师模型的大量知识来保持生成图像的质量和一致性

ADD 模型具有以下优势：

- **高速生成**：仅需 1-4 步采样即可生成高质量图像，显著减少了生成时间，适用于实时应用。
- **高质量图像**：通过结合对抗性损失和蒸馏损失，生成的图像在细节和逼真度上优于现有的快速生成模型，如单步 GAN 和一些少步扩散模型。
- **灵活性**：支持进一步的多步采样，从而在单步生成的基础上通过迭代增强图像细节。


<h2 id="q-064a">面试问题：SDXL Lightning、DMD2、Hyper-SD 等新一代少步蒸馏方法相比 SDXL-Turbo 有哪些进步？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

SDXL-Turbo（ADD）开启了 SDXL 的少步生成时代，但也存在「单步质量上限有限、CFG 不可用、画风偏写实」等问题。**SDXL Lightning、DMD / DMD2、Hyper-SD** 等新一代蒸馏方法在「质量、可控性、训练稳定性」上做了系统性升级，是当前少步生成的主流。

#### 1. 主流少步蒸馏方法对比

<div align="center">

| 方法 | 提出方 | 核心思想 | 步数 | 是否支持 CFG | 训练稳定性 |
| --- | --- | --- | --- | --- | --- |
| **SDXL-Turbo (ADD)** | Stability AI | 对抗 + 蒸馏，像素空间判别器 | 1 ~ 4 | 不支持（已固化） | 中 |
| **SDXL Lightning** | ByteDance | **渐进式蒸馏 + 对抗 + 多步组合**，支持 1/2/4/8 步多档 | 1 ~ 8 | 受限 | 高，多档统一 |
| **DMD（Distribution Matching Distillation）** | MIT / Adobe | **分布匹配蒸馏**，让学生模型的输出分布逼近教师 | 1 ~ 4 | 不支持 | 中 |
| **DMD2** | MIT / Adobe | DMD 升级版：双判别器 + GAN loss 稳定化 | 1 ~ 4 | 不支持 | 高，画质显著提升 |
| **Hyper-SD** | ByteDance | **轨迹分段一致性蒸馏 + 人类反馈对齐 + LoRA 形式发布** | 1 ~ 8 | 部分支持 | 高，可作为 LoRA 适配器 |
| **PCM（Phased Consistency Model）** | 上海 AI Lab | **分相位的一致性蒸馏**，缓解一致性模型在 SDXL 上的画质塌缩 | 1 ~ 8 | 受限 | 高 |

</div>

#### 2. 相比 SDXL-Turbo 的关键进步

1. **多步档位统一**：SDXL Lightning、Hyper-SD 用一份权重支持 1 / 2 / 4 / 8 步推理，而 Turbo 通常一档一个权重，部署成本大幅下降；
2. **质量显著提升**：DMD2、Hyper-SD 在 4 步推理上的 FID / 视觉质量已经接近教师 SDXL 25 步的水平，远超 SDXL-Turbo；
3. **画风更通用**：Turbo 偏写实、对动漫 / 二次元支持差；Lightning / Hyper-SD 等保留了原始 SDXL 的画风分布；
4. **LoRA 形式发布**：Hyper-SD、Lightning 都提供 LoRA 权重，可以叠加在自定义 SDXL 模型上，而不破坏用户已有的 LoRA 生态；
5. **训练稳定性提升**：DMD2 的双判别器、Hyper-SD 的轨迹分段一致性、PCM 的分相位一致性都解决了「对抗训练塌缩」的工程难题；
6. **部分支持 CFG**：Hyper-SD 支持有限 CFG（小尺度），缓解了 Turbo「CFG 失效」的可控性问题；
7. **配套 reward model 与人类反馈**：Hyper-SD 显式引入了人类偏好对齐，是 SDXL 蒸馏方法中第一个引入 RLHF 思路的。

#### 3. 工程选择建议

- **要求最快、可接受质量妥协**：SDXL-Turbo / DMD（1 步）；
- **追求质量与速度均衡**：SDXL Lightning 4 步 / Hyper-SD 4-8 步；
- **要叠在已有 SDXL fine-tune / LoRA 上**：Hyper-SD（LoRA 形式）；
- **追求新一代最佳画质**：DMD2（4 步）。

#### 4. 跨周期价值

少步蒸馏方法是「**扩散模型从研究走向实时应用**」的关键技术路线；**ADD → DMD → Lightning → Hyper-SD → DMD2** 这条主线后续被 FLUX、SD 3 全部继承，理解 SDXL 上的演进有助于看懂 FLUX-schnell、SD3-Turbo、SD 3.5-Turbo 等新一代 Turbo 模型。

**面试金句**：SDXL-Turbo 解决了「能不能 4 步出图」，新一代 Lightning / DMD2 / Hyper-SD 解决的是「**能不能在 4 步出图同时保留画风、支持 CFG、用 LoRA 发布、训练稳定**」；这是「**研究 demo → 工业可用**」的工程化跃迁。


<h2 id="q-065">面试问题：什么是 SDXL Refiner？</h2>

SDXL Refiner是Stability AI推出的图像精细化模型，作为SDXL生态系统的第二阶段，专门负责提升图像细节质量。它采用了"专家集成"的设计理念：Base模型生成基础结构，Refiner模型优化细节表现。

<div align="center"><img src="./imgs/pipeline.png" alt="SDXL Base + Refiner 两阶段流程" /></div>

#### 核心工作原理

##### 两种使用方式

1. **标准流程**：Base模型完成80%去噪 → Refiner完成剩余20%精细化
2. **SDEdit流程**：Base生成完整图像 → Refiner使用img2img技术优化

#### 技术特点

- **双文本编码器**：OpenCLIP-ViT/G + CLIP-ViT/L，提供更好的语义理解
- **专门优化**：针对低噪声水平的去噪过程进行特殊训练
- **参数规模**：6.06B参数，专注于细节增强

#### 性能提升

根据官方评测，SDXL Base + Refiner的组合相比之前版本：

- 用户偏好度达到91%（远超SD 1.5/2.1）

- 细节清晰度提升约20-30%

- 整体图像质量显著改善

- SDXL Refiner通过专门的精细化设计，成功解决了AI图像生成中的细节问题。它与Base模型的配合使用，让SDXL成为目前最优秀的开源图像生成方案之一。对于追求高质量图像输出的用户，Refiner是不可或缺的工具。


<h1 id="q-066">6.Stable Diffusion 3 有哪些创新点？</h1>

<h2 id="q-067">面试问题：介绍一下 Stable Diffusion 3 的整体架构。与 Stable Diffusion XL 相比，SD 3 的核心架构优化有哪些？详细分析改进意图（VAE、Backbone、Text Encoder）</h2>

Rocky认为Stable Diffusion 3的价值和传统深度学习时代的“YOLOv4”一样，在AIGC时代的工业界、应用界、竞赛界以及学术界，都有非常大的学习借鉴价值，以下是SD 3相比之前系列的改进点汇总：

1. 使用多模态DiT作为扩散模型核心：多模态DiT（MM-DiT）将图像的Latent tokens和文本的tokens拼接在一起，并采用两套独立的权重处理，但是在进行Attention机制时统一处理。
2. 改进VAE：通过增加VAE通道数来提升图像的重建质量。
3. 3个文本编码器：SD 3中使用了三个文本编码器，分别是CLIP ViT-L（参数量约124M）、OpenCLIP ViT-bigG（参数量约695M）和T5-XXL encoder（参数量约4.7B）。
4. 采用优化的Rectified Flow：采用Rectified Flow来作为SD 3的采样方法，并在此基础上通过对中间时间步加权能进一步提升效果。
5. 采用QK-Normalization：当模型变大，而且在高分辨率图像上训练时，attention层的attention-logit（Q和K的矩阵乘）会变得不稳定，导致训练出现NAN，为了提升混合精度训练的稳定性，MM-DiT的self-attention层采用了QK-Normalization。
6. 多尺寸位置编码：SD 3会先在256x256尺寸下预训练，再以1024x1024为中心的多尺度上进行微调，这就需要MM-DiT的位置编码需要支持多尺度。
7. timestep schedule进行shift：对高分辨率的图像，如果采用和低分辨率图像的一样的noise schedule，会出现对图像的破坏不够的情况，所以SD 3中对noise schedule进行了偏移。
8. 强大的模型Scaling能力：SD 3中因为核心使用了transformer架构，所以有很强的scaling能力，当模型变大后，性能稳步提升。
9. 训练细节：数据预处理（去除离群点数据、去除低质量数据、去除NSFW数据）、图像Caption精细化、预计算图像和文本特征、Classifier-Free Guidance技术、DPO（Direct Preference Optimization）技术


#### Stable Diffusion 3的VAE部分的创新

**VAE（变分自编码器，Variational Auto-Encoder）模型在Stable Diffusion 3（SD 3）中依旧是不可或缺的组成部分**，Rocky相信不仅在SD 3模型中，在AIGC时代的未来发展中VAE模型也会持续发挥价值。

到目前为止，在AI绘画领域中关于VAE模型我们可以明确的得出以下经验：

1. VAE作为Stable Diffusion 3的组成部分在AI绘画领域持续繁荣，是VAE模型在AIGC时代中最合适的位置。
2. VAE在AI绘画领域的主要作用，不再是生成能力，而是辅助SD 3等AI绘画大模型的**压缩和重建能力**。
3. **VAE的编码和解码功能，在以SD 3为核心的AI绘画工作流中有很强的兼容性、灵活性与扩展性**，也为Stable Diffusion系列模型增添了几分优雅。

和之前的系列一样，在SD 3中，VAE模型依旧是将像素级图像编码成Latent特征，不过由于SD 3的扩散模型部分全部由Transformer架构组成，所以还需要将Latent特征转换成Patches特征，再送入扩散模型部分进行处理。

之前SD系列中使用的VAE模型是将一个 $H \times W \times 3$ 的图像编码为 $\frac{H}{8} \times \frac{W}{8} \times d$ 的Latent特征，在8倍下采样的同时设置 $d=4$ （通道数），这种情况存在一定的压缩损失，产生的直接影响是对Latent特征重建时容易产生小物体畸变（比如人眼崩溃、文字畸变等）。

所以SD 3模型通过提升 $d$ 来增强VAE的重建能力，提高重建后的图像质量。下图是SD 3技术报告中对不同 $d$ 的对比实验：

<div align="center"><img src="./imgs/SD3中VAE的通道数（channel）消融实验.png" alt="SD 3中VAE的通道数（channel）消融实验" /></div>

我们可以看到，当设置 $d=16$ 时，VAE模型的整体性能（FID指标降低、Perceptual Similarity指标降低、SSIM指标提升、PSNR指标提升）比 $d=4$ 时有较大的提升，所以SD 3确定使用了 $d=16$ （16通道）的VAE模型。

与此同时，随着VAE的通道数增加到16，扩散模型部分（U-Net或者DiT）的通道数也需要跟着修改（修改扩散模型与VAE Encoder衔接的第一层和与VAE Decoder衔接的最后一层的通道数），虽然不会对整体参数量带来大的影响，但是会增加任务整体的训练难度。**因为当通道数从4增加到16，SD 3要学习拟合的内容也增加了4倍**，我们需要增加整体参数量级来提升**模型容量（model capacity）**。下图是SD 3论文中模型通道数与模型容量的对比实验结果：

<div align="center"><img src="./imgs/SD3模型容量和VAE通道数之间的关系.png" alt="SD 3模型容量和VAE通道数之间的关系" /></div>

当模型参数量小时，16通道VAE的重建效果并没有比4通道VAE的要更好，当模型参数量逐步增加后，16通道VAE的重建性能优势开始展现出来，**当模型的深度（depth）增加到22时，16通道的VAE的性能明显优于4通道的VAE**。

不过上图中展示了8通道VAE在FID指标上和16通道VAE也有差不多的效果，Rocky认为在生成领域，只使用一个指标来评价模型整体效果是不够全面的，并且FID只是图像质量的一个间接评价指标，并不能反映图像细节的差异。从重建效果上看，16通道VAE应该有更强的重建性能，而且当模型参数量级增大后，SD 3模型的整体性能上限也大幅提升了，带来了更多潜在的优化空间。

**下面是Rocky梳理的Stable Diffusion 3 VAE完整结构图**，大家可以感受一下其魅力。希望能让大家对这个在Stable DIffusion系列中持续繁荣的模型有一个更直观的认识，在学习时也更加的得心应手：

<div align="center"><img src="./imgs/Stable-Diffusion-3-VAE完整结构图.png" alt="Stable Diffusion 3 VAE完整结构图" /></div>

#### Stable Diffusion 3的Text Encoder部分的创新

作为当前最强的AI绘画大模型之一，Stable Diffusion 3模型都是AIGC算法岗面试中的必考内容。接下来，Rocky将带着大家深入浅出讲解Stable Diffusion 3模型的Text Encoder部分是如何改进的。

Stable Diffusion 3的文字渲染能力很强，同时遵循文本Prompts的图像生成一致性也非常好，**这些能力主要得益于SD 3采用了三个Text Encoder模型**，它们分别是：

1. CLIP ViT-L（参数量约124M）
2. OpenCLIP ViT-bigG（参数量约695M）
3. T5-XXL Encoder（参数量约4.76B）

在SD系列模型的版本迭代中，Text Encoder部分一直在优化增强。一开始SD 1.x系列的Text Encoder部分使用了CLIP ViT-L，在SD 2.x系列中换成了OpenCLIP ViT-H，到了SDXL则使用CLIP ViT-L + OpenCLIP ViT-bigG的组合作为Text Encoder。有了之前的优化经验，SD 3更进一步增加Text Encoder的数量，加入了一个参数量更大的T5-XXL Encoder模型。

与SD模型的结合其实不是T5-XXL与AI绘画领域第一次结缘，早在2022年谷歌发布Imagen时，就使用了T5-XXL Encoder作为Imagen模型的Text Encoder，**并证明了预训练好的纯文本大模型能够给AI绘画大模型提供更优良的文本特征**。接着OpenAI发布的DALL-E 3也采用了T5-XXL Encoder来提取文本（Prompts）的特征信息，足以说明T5-XXL Encoder模型在AI绘画领域已经久经考验。

**这次SD 3加入T5-XXL Encoder也是其在文本理解能力和文字渲染能力大幅提升的关键一招**。Rocky认为在AIGC时代，随着各细分领域大模型技术的持续繁荣，很多灵感创新都可以在AI绘画领域中迁移借鉴与应用，从而推动AI绘画大模型的持续发展与升级！

总的来说，**SD 3一共需要提取输入文本的全局语义和文本细粒度两个层面的信息特征**。

首先需要**提取CLIP ViT-L和OpenCLIP ViT-bigG的Pooled Text Embeddings，它们代表了输入文本的全局语义特征**，维度大小分别是768和1280，两个embeddings拼接（concat操作）得到2048的embeddings，然后经过一个MLP网络并和Timestep Embeddings相加（add操作）。

接着我们需要**提取输入文本的细粒度特征**。这里首先分别提取CLIP ViT-L和OpenCLIP ViT-bigG的倒数第二层的特征，拼接在一起得到77x2048维度的CLIP Text Embeddings；再从T5-XXL Encoder中提取最后一层的T5 Text Embeddings特征，维度大小是77x4096（这里也限制token长度为77）。紧接着对CLIP Text Embeddings使用zero-padding得到和T5 Text Embeddings相同维度的编码特征。最后，将padding后的CLIP Text Embeddings和T5 Text Embeddings在token维度上拼接在一起，得到154x4096维度的混合Text Embeddings。这个混合Text Embeddings将通过一个linear层映射到与图像Latent的Patch Embeddings特征相同的维度大小，最终和Patch Embeddings拼接在一起送入MM-DiT中。具体流程如下图所示：

<div align="center"><img src="./imgs/SD3中TextEncoder注入和融合文本特征的示意图.png" alt="SD 3中Text Encoder注入和融合文本特征的示意图" /></div>

虽然SD 3采用CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL Encoder的组合带来了文字渲染和文本一致性等方面的效果增益，但是也限制了T5-XXL Encoder的能力。因为CLIP ViT-L和OpenCLIP ViT-bigG都只能默认编码77 tokens长度的文本，这让原本能够编码512 tokens的T5-XXL Encoder在SD 3中也只能处理77 tokens长度的文本。而SD系列的“友商”模型DALL-E 3由于只使用了T5-XXL Encoder一个语言模型作为Text Encoder模块，所以可以输入512 tokens的文本，从而发挥T5-XXL Encoder的全部能力。

更多详细内容，大家可以查阅：[深入浅出完整解析Stable Diffusion 3（SD 3）和FLUX.1系列核心基础知识](https://zhuanlan.zhihu.com/p/684068402)


<h2 id="q-067a">面试问题：MM-DiT 与原始 DiT 的核心差异是什么？为什么 SD 3 选择 MM-DiT 而不是直接复用 DiT？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

DiT（Diffusion Transformer，W. Peebles & S. Xie, ICCV 2023）是把扩散模型的 Backbone 从 U-Net 换成 Transformer 的奠基工作；MM-DiT（Multimodal DiT, SD 3）则是 DiT 在「文本-图像联合建模」上的关键升级，是 SD 3、FLUX.1 共同的 Backbone 设计模板。

#### 1. 原始 DiT 的核心结构

- 输入只有图像 latent token；
- 文本条件通过 **AdaLN-Zero**（Adaptive LayerNorm 调制 scale / shift）注入到 Transformer 的每一个 block；
- 文本条件被压缩成一个 pooled embedding 后经 MLP 给出 scale / shift；
- 文本与图像之间**不存在 token 级 attention 交互**，文本只能通过「全局调制」影响图像。

这种范式在「类别条件生成」上效果好（如 ImageNet class-conditional），但**对长 prompt、复杂语义、文字渲染不够友好**——文本的细粒度信息在 pooling 中被丢失。

#### 2. MM-DiT 的核心改进

<div align="center">

| 维度 | 原始 DiT | MM-DiT |
| --- | --- | --- |
| **输入序列** | 仅图像 token | **图像 token + 文本 token 拼接成统一序列** |
| **文本注入方式** | AdaLN-Zero 全局调制 | **Self-Attention 内文本与图像 token 双向交互** |
| **权重共享** | 单套 Transformer 权重 | **图像与文本各自一套独立 Linear / FFN 权重**，但共享 Attention |
| **跨模态交互粒度** | 全局 | **Token 级、双向** |
| **文字渲染能力** | 弱 | **强**（双向 attention 让模型知道每个文字 token 应该出现在哪个图像 token） |
| **长 prompt 利用** | 有限 | 显著增强 |

</div>

#### 3. 关键技术细节

- **双权重单注意力（Dual-Stream → Single Attention）**：图像和文本 token 各自先经过自己的 Linear / FFN（参数不共享），再在 Self-Attention 中拼接序列做联合注意力，输出后再分回各自分支；这种「两路独立、共用 attention」是 MM-DiT 的标志性设计；
- **QK-Norm**：对 Q、K 做 L2 norm，缓解高分辨率训练时 attention logits 爆炸 / NaN 的问题；
- **AdaLN-Zero 仍然保留**：用于注入 timestep + Pooled Text Embedding 作为全局调制；
- **Token 拼接顺序**：通常 `[text_tokens, image_tokens]`，attention mask 全开；
- **位置编码**：图像走 2D RoPE / 2D positional embedding；文本走 1D positional embedding；统一序列内坐标各自独立。

#### 4. 为什么 SD 3 选择 MM-DiT 而非 DiT

1. **文字渲染需求**：DALL-E 3、Imagen 已经证明强文本对齐与文字渲染需要文本与图像 token 级交互；DiT 的全局调制做不到；
2. **多文本编码器组合**：SD 3 用了 CLIP-L、OpenCLIP-bigG、T5-XXL 三个编码器，文本 token 数量远大于一个 pooled vector，必须有 token 级注入通道；
3. **多模态扩展性**：未来扩展到「文本 + 参考图 + 深度 + 姿态」多条件生成，MM-DiT 的拼接式注入可以无缝扩展为更多模态序列；
4. **保留 DiT 的 Scaling Law**：MM-DiT 仍然是纯 Transformer，沿用 DiT 的良好 scaling 性质。

**面试金句**：DiT 把扩散 Backbone 从 U-Net 升级为 Transformer，但仍把文本当作「全局调制」；**MM-DiT 把文本和图像放进同一个 token 序列做双向 self-attention，配合「双权重 + 单注意力」的设计**，让 SD 3 / FLUX 在长 prompt、文字渲染、多模态扩展上获得了 DiT 无法达到的能力。


<h2 id="q-067b">面试问题：SD 3 的 Rectified Flow 训练目标相比 ε-prediction 的本质差别是什么？给少步采样带来了哪些工程优势？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

Rectified Flow（RF）是 SD 3、FLUX.1 共同采用的训练目标，是相对 DDPM 的 ε-prediction 在「数学路径 + 采样效率」上的双重升级。

#### 1. 两者的训练目标对比

**ε-prediction（DDPM）**：

- 数据 → 噪声的过程是带噪声的随机扩散：$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$
- 网络预测加入的噪声 $\epsilon$；
- 反向过程是马尔可夫链，沿弯曲路径回到数据。

**Rectified Flow**：

- 数据 → 噪声的路径直接定义为**线性插值**：

  ```math
  x_t = (1 - t)\, x_0 + t\, \epsilon,\quad t \in [0, 1]
  ```

- 网络预测**速度场** $v_t = \epsilon - x_0$（与 v-prediction 形式一致，但是连续时间）；
- 训练目标：

  ```math
  \mathcal{L}_{\text{RF}} = \mathbb{E}_{t, x_0, \epsilon}\bigl\|v_\theta(x_t, t) - (\epsilon - x_0)\bigr\|^2
  ```

- 反向过程是常微分方程（ODE）：

  ```math
  \frac{dx}{dt} = v_\theta(x_t, t)
  ```

#### 2. 本质差别

<div align="center">

| 维度 | ε-prediction | Rectified Flow |
| --- | --- | --- |
| 路径类型 | 弯曲（DDPM 噪声调度决定） | **直线**（数据 ↔ 噪声线性插值） |
| 训练目标 | 噪声 $\epsilon$ | **速度场** $v = \epsilon - x_0$ |
| 反向过程 | 随机马尔可夫链 / DDIM ODE | 纯 ODE |
| Loss 量级随 $t$ 分布 | 高 $t$ 易学、低 $t$ 数值不稳 | **全 $t$ 均衡** |
| 少步采样难度 | 高（弯曲路径需要多步） | **低**（直线路径少步即可逼近） |
| 是否便于二次蒸馏 | 一般 | **极好**（直线路径天然适合 reflow / 一致性蒸馏） |

</div>

#### 3. 给少步采样带来的工程优势

1. **直线路径 → 少步精度高**：数据到噪声的最优 transport 路径在理想情况下是直线；RF 直接用线性插值定义路径，让网络学会「沿直线方向走」，因此少步 Euler 采样误差小；
2. **天然适配 v-prediction 范式**：v 在所有 $t$ 上 loss 量级均衡，模型在低噪、高噪段都能学习；
3. **Reflow 二次蒸馏**：RF 论文最有价值的训练技巧——把已经训练好的 RF 模型生成的 noise-data 配对再训练一遍，可以**进一步把弯曲路径拉直**，从而 1~4 步采样质量大幅提升（FLUX.1-schnell 的 4 步推理就是这条路线）；
4. **采样器简化**：RF 的反向过程是纯 ODE，可以直接用 Euler、Midpoint、RK4 等通用 ODE 求解器，无需 DDIM / DPM-Solver / UniPC 等扩散专用采样器；
5. **timestep schedule 更直观**：RF 的 $t \in [0, 1]$ 直接表示「数据到噪声的进度」，比 DDPM 的离散 $t \in \{1,\dots,T\}$ 更易于做 lognorm shift 等噪声调度优化；
6. **训练效率更高**：SD 3 论文报告，相同算力下 RF 的 FID 收敛速度优于 ε-prediction。

#### 4. 工程实践中的注意点

- **timestep schedule shift**：高分辨率训练时仍需对 $t$ 做 lognorm 偏移（见 [SD 3 / 3.5 timestep schedule shift](#q-068a)）；
- **Sampler 默认走 Euler**：FLUX、SD 3 在 diffusers 里默认是 `FlowMatchEulerDiscreteScheduler`；
- **CFG 仍然有效**：RF 与 CFG 完全兼容，CFG 公式形式不变；
- **不能直接复用 DDPM 的预训练权重**：训练目标不同，权重不通用，需要从头训或用 RF 重训。

**面试金句**：RF 把数据-噪声路径**显式定义为直线**，把网络从「预测噪声」升级为「预测速度场」，让**少步采样精度**与**二次蒸馏（reflow）**两项工程能力都成为天然属性；这是 SD 3 / FLUX 在 4 步出图质量上跨越式提升的根本原因。


<h2 id="q-068">面试问题：Stable Diffusion 3 中使用的训练方法有哪些创新点？</h2>


<h2 id="q-068a">面试问题：SD 3 / SD 3.5 在高分辨率训练中对 timestep schedule 做的 shift 具体是怎么做的？为什么对大尺寸训练至关重要？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐ (4/5)**

SD 3 论文中明确提出，在做大分辨率训练（如 1024×1024 及以上）时，必须对 Rectified Flow 的 timestep schedule 做 **shift（偏移）**，否则模型在高分辨率下会出现「破坏不够」「低频结构残留」的现象。这是 SD 3、SD 3.5、FLUX.1 共享的关键训练技巧。

#### 1. 为什么高分辨率训练需要 shift schedule

- **加噪过程的「破坏强度」与分辨率耦合**：在固定 noise schedule 下，高分辨率图像在同一 $t$ 上的「相对破坏程度」更弱——因为高分辨率图像有更多低频信号，相同方差的高斯噪声只能盖住高频，低频结构仍清晰可辨；
- 这使得在高 $t$ 区间（接近纯噪声端）模型仍能看到原图的低频骨架，**学不到「从纯噪声开始生成」的能力**；
- 表现为：高分辨率推理时初始几步生成出的图像「结构空泛 / 色调单一」，后期采样很难纠正。

#### 2. SD 3 的 timestep shift 公式

SD 3 论文给出的 logit-normal + shift 公式（针对 RF 的 $t \in [0, 1]$）：

```math
t' = \frac{m \cdot t}{1 + (m - 1) \cdot t}
```

其中 $m$ 是 shift 系数，$m > 1$ 把 $t$ 整体推向更接近 1（更接近纯噪声）的区间；分辨率越高，建议的 $m$ 越大。

SD 3 论文实验得到的经验值（base 训练分辨率 1024×1024）：

<div align="center">

| 训练分辨率 | 推荐 shift $m$ |
| --- | --- |
| 256×256 | 1.0（无 shift） |
| 512×512 | ≈ 1.5 |
| 1024×1024 | ≈ 3.0 |
| 2048×2048 | ≈ 6.0 |

</div>

#### 3. 对训练 / 推理两端的影响

- **训练端**：采样 $t$ 时按 shift 后的分布采样，更多样本落在高 $t$ 区，使模型在「接近纯噪声 → 数据」的最关键阶段获得足够的训练信号；
- **推理端**：采样器使用同样 shift 后的离散 timestep 序列；diffusers 中 `FlowMatchEulerDiscreteScheduler` 提供 `shift` 参数；
- **少步采样兼容性**：少步推理（FLUX.1-schnell 的 4 步、SD 3-Turbo）尤其依赖正确的 shift——shift 错了少步出图直接塌缩；
- **训练 / 推理 shift 必须一致**：训练 m=3、推理 m=1 会出现严重的画质降级。

#### 4. 跨周期价值

timestep schedule shift 不只对 SD 3 / FLUX 有效；它揭示了一个**普适规律**：随着扩散模型分辨率提升，需要重新设计 noise schedule，让加噪过程在视觉上「真正破坏图像」。这一思路在 EDM2、Karras 系列、Cosmos、视频生成模型中都有相似的体现。

**面试金句**：高分辨率图像低频信号更强，固定 noise schedule 在高 $t$ 处「破坏不够」，模型学不到从纯噪声起步的能力；SD 3 用 shift 公式把 timestep 偏向高噪声端，让训练 / 推理 / 少步采样都获得正确的噪声水平分布。这是 SD 3、SD 3.5、FLUX.1 在 1024+ 分辨率下能稳定训练并少步出图的关键工程细节。


<h2 id="q-069">面试问题：训练 Stable Diffusion 过程中官方使用了哪些训练技巧？</h2>


<h2 id="q-070">面试问题：介绍一下 Stable Diffusion 3.5 系列的原理</h2>


<h2 id="q-071">面试问题：为什么 Stable Diffusion 3 使用三个文本编码器？</h2>

Stable Diffusion 3作为一款先进的文本到图像模型,采用了三重文本编码器的方法。这一设计选择显著提升了模型的性能和灵活性。

<div align="center"><img src="./imgs/sd3pipeline.png" alt="Stable Diffusion 3 Pipeline 示意图" /></div>

#### 1. 三个文本编码器

Stable Diffusion 3使用以下三个文本编码器:

1. CLIP-L/14
2. CLIP-G/14
3. T5 XXL

#### 2. 使用多个文本编码器的原因

##### 2.1 提升性能

使用多个文本编码器的主要动机是提高整体模型性能。通过组合不同的编码器,模型能够捕捉更广泛的文本细微差别和语义信息,从而实现更准确和多样化的图像生成。

##### 2.2 推理时的灵活性

多个文本编码器的使用在推理阶段提供了更大的灵活性。模型可以使用三个编码器的任意子集,从而在性能和计算效率之间进行权衡。

##### 2.3 通过dropout增强鲁棒性

在训练过程中,每个编码器都有46.3%的独立dropout率。这种高dropout率鼓励模型从不同的编码器组合中学习,使其更加鲁棒和适应性强。

#### 3. 各个编码器的影响

##### 3.1 CLIP编码器(CLIP-L/14和OpenCLIP-G/14)

- 这些编码器对大多数文本到图像任务至关重要。
- 它们在广泛的提示范围内提供强大的性能。

##### 3.2 T5 XXL编码器

- 虽然对复杂提示很重要,但其移除的影响较小:
  - 对美学质量评分没有影响(人类偏好评估中50%的胜率)
  
  - 对提示遵循性有轻微影响(46%的胜率)
  
  - 对生成书面文本的能力有显著贡献(38%的胜率)
  
    （胜率是完整版对比其他模型的效果，下图是对比其他模型以及不使用T5的sd3的胜率图）
  
    <div align="center"><img src="./imgs/sd3实验.png" alt="SD 3 文本编码器消融实验对比" /></div>

#### 3.3 实际应用

1. **内存效率**: 用户可以在大多数提示中选择排除T5 XXL编码器(拥有47亿参数),而不会造成显著的性能损失,从而节省大量显存。

2. **任务特定优化**: 对于涉及复杂描述或大量书面文本的任务,包含T5 XXL编码器可以提供明显的改进。

3. **可扩展性**: 多编码器方法允许在模型的未来迭代中轻松集成新的或改进的文本编码器。


<h2 id="q-072">面试问题：Stable Diffusion 3 中数据标签工程的具体流程是什么样的？</h2>

**目前AI绘画大模型存在一个很大的问题是模型的文本理解能力不强**，主要是指AI绘画大模型生成的图像和输入文本Prompt的一致性不高。举个例子，如果说输入的文本Prompt非常精细复杂，那么生成的图像内容可能会缺失这些精细的信息，导致图像与文本的内容不一致。这也是AI绘画大模型Prompt Following能力的体现。

产生这个问题归根结底还是由训练数据集本身所造成的，**更本质说就是图像Caption标注太过粗糙**。

SD 3借鉴了DALL-E 3的数据标注方法，使用**多模态大模型CogVLM**来对训练数据集中的图像生成高质量的Caption标签。

**目前来说，DALL-E 3的数据标注方法已经成为AI绘画领域的主流标注方法，很多先进的AI绘画大模型都使用了这套标签精细化的方法**。

这套数据标签精细化方法的主要流程如下：

1. 首先整理数据集和对应的原始标签。
2. 接着使用CogVLM多模态大模型对原始标签进行优化扩写，获得长Caption标签。
3. 在SD 3的训练中使用50%的长Caption标签+50%的原始标签混合训练的方式，提升SD 3模型的整体性能，同时标签的混合使用也是对模型进行正则的一种方式。

具体效果如下所示：

<div align="center"><img src="./imgs/SD3数据标注工程.png" alt="SD 3数据标注工程" /></div>


<h2 id="q-073">面试问题：SD3-Turbo 用的蒸馏方法是什么？</h2>

论文链接:[2403.12015](https://arxiv.org/pdf/2403.12015)

#### 方法结构

论文提出了一种新的蒸馏方法——**潜在对抗扩散蒸馏（Latent Adversarial Diffusion Distillation, LADD）**，用于将大规模的扩散模型高效地蒸馏成快速生成高分辨率图像的模型。该方法主要用于基于**Stable Diffusion 3**的优化，目标是生成多比例、高分辨率的图像。与传统的对抗扩散蒸馏（ADD）方法不同，LADD直接在潜在空间（latent space）中进行训练，从而减少了内存需求，并避免了从潜在空间解码到像素空间的昂贵操作。其整体架构包括以下几个关键组件：

1. **生成器（Teacher Model）**：用于生成潜在空间的表示，以进行合成数据的生成。
2. **学生模型（Student Model）**：学习生成器在潜在空间中的分布，以实现快速生成。
3. **判别器（Discriminator）**：用于区分学生模型生成的图像和真实图像的潜在表示，通过对抗训练优化学生模型。

<div align="center"><img src="./imgs/SD3Turbo.jpg" alt="SD3-Turbo LADD 蒸馏方法结构示意图" /></div>

LADD（潜在对抗扩散蒸馏）与ADD（对抗扩散蒸馏）有几个关键区别，主要体现在训练方式、判别器的使用以及生成流程的简化上：

1. **潜在空间训练**：LADD直接在潜在空间（latent space）进行蒸馏，而ADD则需要将图像解码到像素空间，以便判别器进行判别。这种在潜在空间中训练的方式，使得LADD的计算需求更少，因为它避免了从潜在空间到像素空间的解码过程，大幅降低了内存和计算成本。
2. **生成器特征作为判别特征**：ADD使用预训练的DINOv2网络来提取判别特征，但这种方式限制了分辨率（最高518×518像素），且不能灵活调整判别器的反馈层次。LADD则直接利用生成器的潜在特征作为判别器的输入，通过控制生成特征中的噪声水平，可以在高噪声时侧重全局结构，在低噪声时侧重细节，达到了更灵活的判别效果。
3. **判别器和生成器的统一**：在LADD中，生成器和判别器是通过生成特征集成的，避免了额外的判别网络。这种方式不仅降低了系统的复杂度，还可以通过调整噪声分布，直接控制图像生成的全局和局部特征。
4. **多长宽比支持**：LADD能够直接支持多长宽比的训练，而ADD由于解码和判别过程的限制，不易实现这一点。因此，LADD生成的图像在各种长宽比下具有较好的适应性。


<h2 id="q-074">面试问题：Stable Diffusion 3 的图像特征和文本特征在训练前缓存策略有哪些优缺点？</h2>

SD 3与之前的版本相比，整体的参数量级大幅增加，这无疑也增加了训练成本，所以官方的技术报告中也**对SD 3训练时冻结（frozen）部分进行了分析**，主要评估了VAE、CLIP-L、CLIP-G以及T5-XXL的显存占用（Mem）、推理耗时（FP）、存储成本（Storage）、训练成本（Delta），如下图所示，T5-XXL的整体成本是最大的：

<div align="center"><img src="./imgs/SD3各个结构的整体成本.png" alt="SD 3各个结构的整体成本" /></div>

**为了减少训练过程中SD 3所需显存和特征处理耗时，SD 3设计了图像特征和文本特征的预计算策略**：由于VAE、CLIP-L、CLIP-G、T5-XXL都是预训练好且在SD 3微调过程中权重被冻结的结构，所以**在训练前可以将整个数据集预计算一次图像的Latent特征和文本的Text Embeddings，并将这些特征缓存下来**，这样在整个SD 3的训练过程中就无需再次计算。同时上述冻结的模型参数也无需加载到显卡中，可以节省约20GB的显存占用。

但是根据机器学习领域经典的“没有免费的午餐”定理，**预计算策略虽然为我们大幅减少了SD 3的训练成本，但是也存在其他方面的代价**。第一点是训练数据不能在训练过程中做数据增强了，所有的数据增强操作都要在训练前预处理好。第二点是预处理好的图像特征和文本特征需要一定的存储空间。第三点是训练时加载这些预处理好的特征需要一定的时间。

整体上看，**其实SD 3的预计算策略是一个空间换时间的技术**。


<h2 id="q-075">面试问题：Stable Diffusion 3.5 的改进在哪里？</h2>

1、**引入 Query-Key 归一化（QK normalization）**：在训练大型 Transformer 模型时，QK 归一化已成为标准实践。SD3.5 也采用了这一技术，以增强模型训练的稳定性并简化后续的微调和开发。

**2、双注意力层设计**：在 MMDiT 结构中，文本和图像两个模态通常共享同一个注意力层。然而，SD3.5 采用了两个独立的注意力层，以更好地处理多模态信息（MMDiT-X）。

<div align="center"><img src="./imgs/mmdit-x.png" alt="MMDiT-X 双注意力层设计" /></div>


<h2 id="q-075a">面试问题：SD 3 / SD 3.5 与 FLUX.1 在架构与训练范式上的继承与差异是什么？</h2>

**难度评分：⭐⭐⭐⭐⭐ (5/5)  |  考察频率：⭐⭐⭐⭐⭐ (5/5)**

FLUX.1 由 Black Forest Labs 团队（包含 Stable Diffusion 原核心作者）发布，与 SD 3 / SD 3.5 在技术血缘上极为相似——本质上是「**MM-DiT + Rectified Flow + 16 通道 VAE + CLIP + T5-XXL**」这条共同技术路线的两个分支。理解二者的继承与差异，是 2025 年 AIGC 算法岗面试的高频考点。

#### 1. 共同的技术血缘

<div align="center">

| 维度 | 共享设计 |
| --- | --- |
| Backbone | MM-DiT 范式（多模态 Transformer，文本与图像 token 联合 self-attention） |
| 训练目标 | Rectified Flow（速度场预测，直线路径） |
| VAE | 16 通道、8x 下采样、L1 + LPIPS + KL + 对抗 |
| Text Encoder | CLIP-L + T5-XXL（FLUX 弱化了 OpenCLIP-bigG 但仍依赖 CLIP-L 与 T5） |
| 噪声调度 | Logit-Normal + shift（与分辨率耦合） |
| 工程优化 | QK-Norm、bf16 / fp8 推理、特征缓存 |

</div>

#### 2. 关键差异

<div align="center">

| 维度 | SD 3 / SD 3.5 | FLUX.1 |
| --- | --- | --- |
| **Backbone 细节** | MM-DiT（SD 3）→ MMDiT-X（SD 3.5：双注意力层） | **Hybrid 设计**：前段 MM-DiT（Double-Stream Block）+ 后段 Single-Stream Block（图像 token 单独 attention） |
| **Text Encoder 组合** | CLIP-L + OpenCLIP-bigG + T5-XXL | CLIP-L + T5-XXL（弱化双 CLIP） |
| **位置编码** | 1D / 2D 可学习 / sinusoidal | **2D RoPE**（旋转位置编码） |
| **分辨率支持** | 多尺寸（256→1024+） | 原生支持任意长宽比 + 高分辨率 |
| **少步蒸馏** | SD3-Turbo（LADD） | **FLUX.1-schnell**：Reflow + 蒸馏，4 步出图 |
| **开源策略** | SD 3 部分商用受限；SD 3.5 多档（Large、Medium、Turbo）开源 | **三档发布**：dev（非商用）、schnell（Apache 2.0）、pro（API 闭源） |
| **生态成熟度** | 基于 SD 历史生态（diffusers / ComfyUI / LoRA） | 后来居上，2024-2025 年成为开源 SOTA，社区生态快速形成 |
| **多模态扩展** | SD 3.5 Large / Medium 主打文生图 | FLUX.1 Kontext（编辑）、FLUX.1 Krea（实时）、FLUX.1 Tools（Fill / Depth / Canny / Redux）、FLUX.2（多图、长上下文） |

</div>

#### 3. FLUX.1 相对 SD 3 的几个关键工程升级

1. **Hybrid Block 设计（Double + Single Stream）**：前 N 个 block 用 Double Stream（图像 / 文本独立权重 + 联合 attention），后 M 个 block 用 Single Stream（只对图像 token 做 self-attention，文本被 pooled），既保留多模态对齐又降低后段算力；
2. **2D RoPE 位置编码**：相比 SD 3 的可学习位置编码，RoPE 在多分辨率、多长宽比泛化上更稳定；
3. **去掉 OpenCLIP-bigG**：仅保留 CLIP-L + T5-XXL，T5 占主导，文本一致性反而更强；
4. **FLUX.1-schnell 的 Reflow 蒸馏**：把 RF 路径再次拉直后做对抗蒸馏，4 步推理质量已经接近 dev 25 步；
5. **更早完整支持 fp8 / 量化**：FLUX 在发布之初就提供官方 fp8 权重与 NF4 / GGUF 量化生态。

#### 4. 在面试中如何快速回答

- **共同点**：MM-DiT + Rectified Flow + 16ch VAE + CLIP + T5 这条「SD 3 范式」是 FLUX.1 的技术起点；
- **不同点**：FLUX.1 用 Hybrid Block + 2D RoPE + 弱化 CLIP-G + 更激进的 fp8/蒸馏，把工程化推到极致；
- **生态意义**：SD 3.5 是 SD 系列的延续与开源补完，FLUX 系列则是这条范式在 2024-2025 年的最强实现，二者**在技术上是亲兄弟、在生态上互相参照**。

**面试金句**：FLUX.1 与 SD 3 共享「**MM-DiT + Rectified Flow + 16ch VAE + T5-XXL**」的范式 DNA；FLUX 在 Backbone（Hybrid Single+Double Stream）、位置编码（2D RoPE）、蒸馏（Reflow + schnell）、量化（fp8 / NF4）上做了更激进的工程化升级，是这条范式当前最完整的工业级实现。


---

<h1 id="ch-flux-01">第二章 FLUX系列核心高频考点</h1>

<h1 id="q-flux-001">1.介绍一下FLUX.1的原理，与Stable Diffusion 3相比有哪些创新点？</h1>

FLUX.1系列模型是基于Stable Diffuson 3进行了升级优化，是目前性能最强的开源AI绘画大模型，其主要的创新点如下所示：

1. FLUX.1系列模型将VAE的通道数扩展至64，比SD3的VAE通道数足足增加了4倍（16）。
2. 目前公布的两个FLUX.1系列模型都是经过指引蒸馏的产物，这样我们就无需使用Classifier-Free Guidance技术，只需要把指引强度当成一个约束条件输入进模型，就能在推理过程中得到带指定指引强度的输出。
3. FLUX.1系列模型继承了Stable Diffusion 3 的噪声调度机制，对于分辨率越高的图像，把越多的去噪迭代放在了高噪声的时刻上。但和Stable Diffusion 3不同的是，FLUX.1不仅在训练时有这种设计，采样时也使用了这种技巧。
4. FLUX.1系列模型中在DiT架构中设计了双流DiT结构和单流DiT结构，同时加入了二维旋转式位置编码 (RoPE) 策略。
5. FLUX.1系列模型在单流的DiT中引入了并行注意力层的设计，注意力层和MLP并联执行，执行速度有所提升。

### VAE部分的改进

**FLUX.1系列中，FLUX.1 VAE架构依然继承了SD 3 VAE的8倍下采样和输入通道数（16）。在FLUX.1 VAE输出Latent特征，并在Latent特征输入扩散模型前，还进行了Pack_Latents操作，一下子将Latent特征通道数提高到64（16 -> 64），换句话说，FLUX.1系列的扩散模型部分输入通道数为64，是SD 3的四倍**。这也代表FLUX.1要学习拟合的内容比起SD 3也增加了4倍，所以官方大幅增加FLUX.1模型的参数量级来提升模型容量（model capacity）。下面是Pack_Latents操作的详细代码，让大家能够更好的了解其中的含义：

```
@staticmethod
def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents
```

**可以看到FLUX.1模型的Latent特征Patch化方法是将 $2\times2$ 像素块直接在通道维度上堆叠。这种做法保留了每个像素块的原始分辨率，只是将它们从空间维度移动到了通道维度。与之相对应的，SD 3使用下采样卷积来实现Latent特征Patch化，但这种方式会通过卷积减少空间分辨率从而损失一定的特征信息。**

Rocky再举一个形象的例子来解释SD 3和FLUX.1的Patch化方法的不同：
1. SD 3（下采样卷积）：想象我们有一个大蛋糕，SD 3的方法就像用一个方形模具，从蛋糕上切出一个 $2\times2$ 的小方块。在这个过程中，我们提取了蛋糕的部分信息，但是由于进行了压缩，Patch块的大小变小了，信息会有所丢失。
2. FLUX.1（通道堆叠）：FLUX.1 的方法更像是直接把蛋糕的 $2\times2$ 块堆叠起来，不进行任何压缩或者切割。我们仍然保留了蛋糕的所有部分，但是它们不再分布在平面上，而是被一层层堆叠起来，像是三明治的层次。这样一来，蛋糕块的大小没有改变，只是它们的空间位置被重新组织了。

总的来说，**相比SD 3，FLUX.1将 $2\times2$ 特征Patch化操作应用于扩散模型之前**。这也表明FLUX.1系列模型认可了SD 3做出的贡献，并进行了继承与优化。

目前发布的FLUX.1-dev和FLUX.1-schnell两个版本的VAE结构是完全一致的。**同时与SD 3相比，FLUX.1 VAE并不是直接沿用SD 3的VAE，而是基于相同结构进行了重新训练，两者的参数权重是不一样的**。并且SD 3和FLUX.1的VAE会对编码后的Latent特征做平移和缩放，而之前的SD系列中VAE仅做缩放：

```
def encode(self, x: Tensor) -> Tensor:
    z = self.reg(self.encoder(x))
    z = self.scale_factor * (z - self.shift_factor)
    return z
```

平移和缩放操作能将Latent特征分布的均值和方差归一化到0和1，和扩散过程加的高斯噪声在同一范围内，更加严谨和合理。

下面是**Rocky梳理的FLUX.1-dev/schnell系列模型的VAE完整结构图**，希望能让大家对这个从SD系列到FLUX.1系列都持续繁荣的模型有一个更直观的认识，在学习时也更加的得心应手：

![FLUX.1-dev/schnell VAE完整结构图](./imgs/FLUX.1-VAE完整结构图.png)

**Rocky认为Stable Diffusion系列和FLUX.1系列中VAE模型的改进历程，为工业界、学术界、竞赛界以及应用界都带来了很多灵感，有很好的借鉴价值。Rocky也相信AI绘画中针对VAE的优化是学术界一个非常重要的论文录用点！**


<h1 id="q-flux-002">2.FLUX.1在训练过程中使用了哪些优化技巧？</h1>

<h2 id="q-flux-003">面试问题：FLUX.1模型的微调训练流程一般包含哪几部分核心内容？</h2>


<h2 id="q-flux-004">面试问题：FLUX.1模型的微调训练流程中有哪些关键参数？</h2>


<h1 id="q-flux-005">3.介绍一下FLUX.1-Lite与FLUX.1的异同</h1>


<h1 id="q-flux-006">4.介绍一下FLUX.1 Kontext的原理，有哪些创新点？</h1>

<h2 id="q-flux-007">面试问题：介绍一下FLUX.1 Kontext的原理，FLUX.1 Kontext能够执行哪些AIGC任务？</h2>


<h1 id="q-flux-008">5.介绍一下FLUX.1 Krea的原理，有哪些创新点？</h1>

<h2 id="q-flux-009">面试问题：介绍一下FLUX.1 Krea的训练策略</h2>


<h2 id="q-flux-010">面试问题：介绍一下FLUX.1-Krea中监督微调（SFT）的流程</h2>

FLUX.1 Krea模型中的监督微调（SFT）是其摆脱“AI感”，生成具有照片级真实感和独特美学图像的关键步骤。

### SFT的流程

FLUX.1 Krea 的SFT流程核心是使用精选数据，将通用的基础模型“调教”成一个具有特定审美品味的模型。

1.  **起点：优质的“原始”基础模型**
    SFT并非从零开始，它需要一个知识渊博的“胚子”。FLUX.1 Krea 使用的是Black Forest Labs提供的 **`flux-dev-raw`基础模型**。同时进行了大规模的预训练，其特点是已经具备了大量的“世界知识”（能理解各种物体、风格和概念），但尚未经过过度微调调优，保留了原始生成分布的多样性，可塑性非常强。

2.  **核心：精心策划的训练数据**
    数据是SFT的灵魂，FLUX.1 Krea 严格遵循 **“质量重于数量”** 的原则。其训练数据主要包括：
    - **高质量图像-文本对**：团队手工挑选了符合其审美标准的高质量图像数据集。
    - **合成样本**：在SFT阶段还加入了来自 **Krea自身模型的高质量合成样本**，这有助于稳定模型在迭代过程中的性能。

    整个SFT阶段使用的数据量可能远少于预训练，但凭借极高的数据质量，足以让模型领会到期望的美学风格。

3.  **关键技术：自定义损失函数与CFG分布微调**
    这是一个技术上的重要细节。由于 `flux-dev-raw` 是一个经过指导蒸馏的模型，FLUX.1 Krea 的团队设计了一个**自定义的损失函数，直接在无分类器引导（CFG）的分布上对模型进行微调**。我们可以这样理解：CFG是生成过程中用于控制图像与提示词相关性的一个技术，直接在此分布上优化，能更有效地引导模型朝着既遵守提示词又具备高美学质量的方向生成图像。

### SFT的原理与作用

从原理上看，SFT的本质是让大模型通过“模仿”来重塑行为。

- **原理**：SFT通过**最小化模型预测与目标答案（高质量图像）之间的交叉熵损失**来进行训练。简单来说，就是通过调整模型参数，让它生成的图像在视觉特征上越来越接近那些精心准备的高质量训练图片，同时忽略训练数据中的缺陷部分。

- **作用**：经过SFT阶段后，FLUX.1 Krea大模型发生了根本性的转变：
    - **确立美学基础**：从能“画出”东西，转变为能画出“好看”的东西，初步建立起对自然光影、真实质感和协调构图的偏好。
    - **提升图像质量**：生成的图像在清晰度、结构准确性和细节丰富度上显著提升。
    - **保留多样性**：由于基础模型的“原始”性和数据的质量，模型在确立风格的同时，并未丧失生成的多样性。

### SFT微调与前后环节的协作

SFT微调并非孤立的环节，它为后续的RLHF阶段打下了坚实的基础。

- **SFT与预训练**：预训练让大模型“见多识广”，学会了世界的各种可能性（包括优质和劣质图像），为SFT阶段的负面提示词等技术提供了基础。SFT则是在此基础上做“减法”和“精修”，引导模型专注于高质量和高美学的部分。

- **SFT与RLHF**：SFT解决了“好看”的问题，而RLHF则进一步解决“更符合人类偏好”的问题。SFT可以被看作是**RLHF的必要准备**，它先让模型具备稳定的高质量输出能力，然后RLHF再通过人类偏好数据对这个能力进行精细校准和微调，使作品的风格更鲜明、更符合特定的艺术标准。

为了让你更清晰地理解SFT在整个过程中的承上启下作用，下表对比了这三个核心阶段：

| **阶段** | **核心目标** | **数据特点** | **对模型的影响** |
| :--- | :--- | :--- | :--- |
| **预训练** | **模式覆盖**与**世界理解** | 海量、多样化的图像-文本对，包含各种质量的图像 | 建立通用视觉知识，最大化生成多样性 |
| **监督微调 (SFT)** | **确立美学基础**与**提升质量** | 少量但极致精选的高质量/合成图像 | 学会生成高清、结构准确、符合特定审美的图像 |
| **RLHF** | **对齐人类偏好**与**风格强化** | 小规模、带有明确艺术导向的人类偏好数据 | 进一步校准输出，使风格更鲜明，更稳健地符合人类审美 |


<h2 id="q-flux-011">面试问题：FLUX.1 Krea的后训练过程中有哪些关键要点？</h2>

在监督微调阶段，FLUX.1 Krea精心筛选构建了一个符合官方审美标准的、最高质量的图像数据集。同时在训练FLUX.1 Krea大模型的过程中，还加入了来自Krea-1模型的高质量合成图像数据，这些图像被用于增强 SFT 阶段的模型训练效果。

由于flux-dev-raw是一个经过指导式蒸馏（guidance distilled）的模型，官方设计了一种自定义损失函数，直接在无分类器引导（CFG）的分布上对模型进行微调训练。在SFT阶段之后，FLUX.1 Krea模型的图像生成质量得到了显著提升。但是要使FLUX.1 Krea模型更加稳健并达到官方所追求的美学效果，还需要进一步的工作：这就是RLHF的用武之地。

在RLHF阶段，官方应用了一种偏好优化技术的变体，称为TPO（Tuned Preference Optimization），以进一步提升FLUX.1 Krea模型的美学质量和风格化水平。官方使用了高质量的内部偏好数据，这些数据经过严格筛选以确保质量。同时在微调过程中还会进行多轮偏好优化，进一步优化FLUX.1 Krea模型生成图像的风格与质量。

在探索各种后训练技术的过程中，Krea官方发现了一些关键要点：

1. **质量比数量重要的多**：我们只需要非常少量的数据（不到100万）就能进行有效的后训练。虽然更大的数据集规模有助于模型的稳定性和减少偏差，但数据的质量才是最为重要的，使用小规模、精心挑选的数据集进行训练，依然可以达到极佳的模型整体效果。使用的偏好标签是由标注人员精心收集的，这些标注人员非常清楚当前模型的局限性、需要改进的领域、优点和缺点。同时确保图像数据集内容足够多样，以获得聚焦且有代表性的标注结果。
2. **采取主观明确的训练方法**：目前有许多开源的偏好数据集，被用于评估测试偏好微调技术。这些数据集对于测试各种技术确实非常有用。然而，如果直接在现有数据集上进行训练，往往会导致一些意想不到的负面影响，例如模型生成的图像会偏向对称、简单的构图；会有模糊和过度柔和的纹理；会出现色彩风格趋于单一的情况；会回归到”AI感”等。

Krea官方认为，在”全局”用户偏好上微调训练的模型在审美质量上并非最优。对于像文本渲染、解剖结构、物体结构和提示词遵循度这样有客观事实依据的目标，数据的多样性和规模确实是很有帮助的。然而，对于像美学质量这样主观的目标，将不同的审美品味混合在一起几乎是相互抵触的。


<h2 id="q-flux-012">面试问题：介绍一下FLUX.1 Krea的后训练过程中使用的Tuned Preference Optimization技术</h2>

在SFT阶段之后，FLUX.1 Krea模型的**图像质量**已经很高了，但**美学风格和鲁棒性**还未达到理想状态。团队发现，使用现有的开源偏好数据集进行优化会导致模型出现“审美中庸”、风格倒退（回归“AI感”）等问题。

TPO正是为了解决这些问题而设计的、一种**高度主观化**的偏好优化技术。

### 一、TPO是什么？

TPO是一种**基于偏好优化的强化学习技术变体**。它的核心思想不是去学习一个“普适的”、“大众的”人类审美，而是**将模型强烈地、有倾向性地对齐到一个非常具体和明确的美学标准上**。

你可以把它理解为模型的“美学特训营”。在这个特训营里，教练（TPO算法）不是教学生“什么样的话大家都爱听”，而是教他“如何成为一名具有独特风格的艺术家”。

### 二、TPO的核心原理

TPO的技术根源来自于像DPO、KTO这样的直接偏好优化算法。这些算法的共同点是**绕过传统RLHF中难以训练的奖励模型，直接利用偏好数据来微调策略模型（即我们要优化的图像生成模型）**。

我们来简单理解一下这个基本原理：

1.  **传统RLHF的痛点**：
    *   需要先训练一个奖励模型来判断人类更喜欢哪张图片。
    *   然后通过强化学习（如PPO）来优化生成模型，以最大化从奖励模型获得的分数。
    *   这个过程非常不稳定、计算成本高，且奖励模型的偏差会直接影响最终效果。

2.  **直接偏好优化的创新**：
    *   它发现，我们可以不训练独立的奖励模型。
    *   而是将问题转化为一个**分类问题**：直接调整生成模型本身的参数，使得它生成“获胜”图片的概率远大于生成“失败”图片的概率。
    *   其损失函数的核心是：**对于一对图片（获胜图片 `X_win` 和失败图片 `X_lose`），优化后模型对 `X_win` 的偏好概率应该大于对 `X_lose` 的偏好概率。**

### 三、TPO的“Tuned”体现在哪里？—— 其关键创新

“Tuned”这个词精准地描述了TPO的精髓——**精心调谐**。它不仅仅是在做偏好优化，而是对整个过程进行了定制化的改进，主要体现在以下几点：

1.  **高度主观与明确的美学导向**
    *   **问题**：开源偏好数据集混合了各种用户的审美，导致模型学习到的是一个“平均审美”，失去了风格棱角。
    *   **TPO的解决方案**：不使用公开数据集，而是**内部收集具有高度一致性的偏好数据**。标注人员完全理解团队想要的艺术方向（如特定的构图、光影、质感）。这使得优化目标非常纯粹和尖锐。

2.  **多轮迭代优化**
    *   官方提到“*在许多情况下，我们应用了多轮的偏好优化*”。这不是一个一次性的过程。
    *   **流程可能是**：`SFT -> TPO Round 1 -> 生成新样本 -> 收集对新样本的偏好 -> TPO Round 2 -> ...`
    *   这种方式允许团队**逐步校准模型**，使其美学风格越来越精确和稳定，不断强化优点、修正缺点。

3.  **对“过度拟合”的重新定义**
    *   在机器学习中，“过度拟合”通常是个贬义词，意味着模型失去了泛化能力。
    *   但在TPO的哲学里，**对于美学这种主观目标，“过度拟合”到一个特定风格上是可取的，甚至是目标**。团队明确指出：“*让模型过度拟合某种特定风格通常是更好且更简单的做法*”。这确保了FLUX.1 Krea输出风格的**鲜明性和一致性**。

4.  **专注于解决SFT后的遗留问题**
    *   SFT解决了“质量”问题，TPO则专注于“风格”和“鲁棒性”。
    *   TPO的偏好数据很可能**特意针对SFT模型表现薄弱或风格不明确的场景**进行收集，例如复杂的构图、特定的色彩搭配、难以渲染的材质等，从而实现精准打击。

### 四、TPO的工作流程

结合以上原理，TPO的一个典型工作流程可以概括为以下步骤：

```mermaid
flowchart TD
    A[SFT模型] --> B[“生成图像对<br>用于偏好标注”]
    B --> C{“高度聚焦的<br>人工偏好标注”}
    C --> D[应用TPO损失函数微调]
    D --> E[新一代模型]
    E -- 多轮迭代--> B
    E --> F[最终模型<br>“过度拟合”到目标风格]
```

### 总结：TPO带来的效果

通过TPO技术，FLUX.1 Krea实现了：

*   **鲜明的艺术风格**：模型输出的图像具有高度可识别的、一致的“Krea风格”，而不是模糊的“大众脸”。
*   **卓越的美学质量**：在团队定义的审美标准下，图像的光影、构图、色彩和质感都达到了极高水平。
*   **强大的鲁棒性**：即使面对具有挑战性的提示词，模型也能稳定地输出符合其美学标准的图像，而不是崩溃或产生“AI感”十足的图片。
*   **用户友好性**：用户无需编写冗长复杂的提示词来“对抗”模型的中庸审美，简单的提示也能得到具有高级美学感的默认输出。

**总而言之，TPO是FLUX.1 Krea成功的关键技术之一。它代表了一种新一代模型优化的理念：从追求“什么都会一点”的通用模型，转向在特定领域或风格上做到极致的“专家型”模型。这种“主观明确”的优化路径，很可能成为未来顶级AI模型竞争的核心。**

<h1 id="q-flux-013">6.与FLUX.1相比，FLUX.2有哪些创新点？</h1>

FLUX.2系列更新的新特性与核心优化亮点，具体如下：
1. **支持参考图生成**：最多可输入 10 张参考图像，在角色、产品及风格一致性上达到当前最佳水平。支持显式图像索引，用户可在提示词中通过编号引用特定图像，例如“将图 2 中的衣服穿在图 1 的角色身上”。
2. **图像细节与照片级真实感**：生成图像具备更丰富的细节、更清晰的纹理与更自然的光照表现，适用于产品摄影、可视化及类似专业摄影场景。
3. **文本渲染能力提升**：可稳定生成复杂排版、信息图表、表情包及含细小文字的 UI 界面模型，支持中文输入与中文文字渲染，已具备生产环境可用性。
4. **增强的提示词遵循**：能够更准确地理解并执行复杂的结构化指令（支持 JSON 格式），包括多部分提示词及构图约束。
5. **丰富的世界知识**：模型在现实世界知识、光照逻辑与空间关系方面表现更加合理，可生成场景更连贯、行为更符合预期的图像。
6. **更高分辨率与灵活的宽高比**：支持最高 4MP（例如 1920×1920）的图像编辑分辨率，并允许灵活的输入与输出比例。
7. **支持十六进制颜色描述**：可通过如 #DDC57A 的十六进制代码精准描述对象颜色，在色彩控制方面表现优异。


### FLUX.2的Text Encoder部分的优化

在Text Encoder部分，FLUX.2的文本编码器不再使用T5和CLIP，而是改用了**Mistral-3-24B视觉语言大模型**（VLM大模型，Mistral-Small-3.2-24B-Instruct-2506），视觉语言大模型提供真实世界知识和上下文理解，增强了对世界、材质、空间关系和构图的建模能力。同时使用单个文本编码器极大地简化了Prompt Embeddings的计算过程。


### FLUX.2的DiT Backbone部分的优化

在DiT Backbone部分，**FLUX.2沿用了与FLUX.1相同的MM-DiT + 并行DiT相结合的整体架构**。简言之，MM-DiT模块首先在独立处理图像潜变量和条件文本，仅在注意力计算环节将二者融合，因此被称为“双流”块。随后的并行DiT模块则对拼接后的图像与文本流进行操作，可视为“单流”块。

从FLUX.1到FLUX.2，DiT架构的核心改进如下：

1. 对DiT部分进行了Scaling，参数量从FLUX.1的12B增加32B。
2. 时间与引导信息（Timestep and Guidance Scale，以 AdaLayerNorm-Zero 调制参数的形式）分别在所有双流块和所有单流块间共享，而非如FLUX.1中为每个块单独设置调制参数，从而降低整体参数量。
3. 模型中所有层均不再使用偏置参数。具体而言，两种变换器块中的注意力子块与前馈子块在其任何层中均未使用偏置参数。
4. 在FLUX.1中，单流变换器块将注意力输出投影与前馈网络输出投影进行了融合。FLUX.2的单流块进一步将注意力QKV投影与前馈网络的输入投影相融合，从而实现了完全并行的Transformer块结构：

![FLUX.2的DiT部分模块示意图](./imgs/FLUX.2的DiT部分模块示意图.png)

需要注意的是，与上图中的 ViT-22B 块相比，FLUX.2 采用了SwiGLU作为多层感知机的激活函数，而非使用GELU激活函数（同时也不使用偏置参数）。

FLUX.2 中单流模块的比例显著提高（双流块与单流块的数量比为 8:48，而 FLUX.1 为 19:38）。这意味着单流模块在 DiT 参数中所占比例更大：FLUX.1-12B 约有 54% 的参数位于双流块中，而 FLUX.2-32B 仅有约 24% 的参数在双流模块内（约 73% 的参数集中在单流模块中）。

最后，FLUX.2 在位置编码设计上也进行了调整。FLUX.1 采用 3D RoPE，其中前两维分别编码图像的宽（w）和高（h），第三维为时间维度 t，在生成时固定为 0；而在 FLUX.1 Kontext 版本中，该 t 值对于输入条件图像设为 1，以区分目标图像与条件图像。

FLUX.2 则升级为 4D RoPE 编码：第一维为 t，用于区分目标图像与条件图像——目标图像（对应噪声潜变量 token）的 t 设为 0，而条件图像的 t 则以 10 为间隔依次递增（如 10、20……）；第二维和第三维仍分别对应图像宽高（w 和 h）；第四维为 l，专门用于编码文本 token 的序列位置，对于图像潜变量则固定为 0。因此，新增的第四维主要作用是为文本 token 赋予位置信息，而此前 FLUX.1 中所有文本 token 的位置编码均固定为 0，并未区分其顺序。


### FLUX.2的VAE部分的优化

在VAE部分，FLUX.2也进行了重要升级。**新版VAE在可学习性、重建质量与压缩率三者间实现了更优的平衡**。

这里的“可学习性”指生成模型（即DiT）在VAE的潜在特征空间（Latent特征空间）中学习生成新样本的难易程度。若潜在特征具备良好的语义化表征，生成任务将更易建模，因为模型仅需捕捉高层语义关系，而无需重构低层感知细节。然而，这种方法可能牺牲图像重建质量，并降低压缩效率。

“质量”在此特指VAE的重建能力，即Decoder解码器能否从压缩后的潜在表示中高保真地还原原始图像。过度压缩通常会引入感知失真与细节丢失。尽管引入感知损失与对抗训练可提升重建效果，但高压缩比往往导致保真度下降。此外，若VAE训练仅追求重建精度，而未对潜在空间施加语义约束，则可能产生含高频噪声或结构混乱的潜在表示，增加生成模型的学习难度。

“压缩率”对应潜在特征的维度，更高的压缩率有助于提升计算效率，但也可能削弱重建质量与生成模型对真实数据分布的拟合能力。

**这三项目标本质上相互制约**：提高压缩率通常会损害重建质量与可学习性；追求完美重建则需降低压缩程度；而为提升语义层面的可学习性，又可能不得不放弃部分底层感知细节。因此，理想的权衡策略是剔除人眼不可感知的信息，同时保留富含语义、利于生成模型高效学习的结构特征——这也正是FLUX.2 VAE的核心设计目标。

相较于FLUX.1 VAE，FLUX.2 VAE在保持重建质量的同时，显著提升了可学习性。具体改进包括：**在保持空间压缩率为8倍的前提下，进一步增加潜在特征的维度（SD-VAE为4维，FLUX.1 VAE为16维，FLUX.2 VAE提升至32维）。潜在维度的增加并未改变DiT处理的token数量，因此不会带来额外计算负担**。此外，在训练过程中引入了语义正则化机制，进一步优化了潜在空间的语义组织结构与可学习性。


---
