# 目录

- [2.ControlNet的模型结构与原理](#2.ControlNet的模型结构与原理)
- [27.CreatiLayout的框架和原理](#27.CreatiLayout的框架和原理)
- [28.Ctrl-X的框架和原理](#28.Ctrl-X的框架和原理)
- [29.OMNIBOOTH的框架和原理](#29.OMNIBOOTH的框架和原理)
- [32.ReCo的框架和原理](#32.ReCo的框架和原理)
- [33.Be Yourself（Bounded Attention for Multi-Subject Text-to-Image Generation）的框架和原理](#33.Be-Yourself（Bounded-Attention-for-Multi-Subject-Text-to-Image-Generation）的框架和原理)
- [34.IFadapter的框架和原理](#34.IFadapter的框架和原理)
- [35.LAW-Diffusion的框架和原理](#35.LAW-Diffusion的框架和原理)
- [36.Check, Locate, Rectify（A Training-Free Layout Calibration System for Text-to-Image Generation）的框架和原理](#36.Check,Locate,Rectify（A-Training-Free-Layout-Calibration-System-for-Text-to-Image-Generation）的框架和原理)
- [37.Conceptrol的框架和原理](#37.Conceptrol的框架和原理)
- [39.Prompt-to-Prompt是什么方法？](#39.Prompt-to-Prompt是什么方法？)
- [40.InstructPix2Pix的训练和推理流程是什么样的？](#40.InstructPix2Pix的训练和推理流程是什么样的？)
- [41.ControlNet系列模型有多少种控制条件？](#41.ControlNet系列模型有多少种控制条件？)
- [42.ControlNet的最小单元是什么样的？](#42.ControlNet的最小单元是什么样的？)
- [43.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？](#43.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？)
- [44.ConrtolNet是如何训练的？](#44.ConrtolNet是如何训练的？)
- [45.ControlNet的损失函数是什么？](#45.ControlNet的损失函数是什么？)
- [46.ControlNet有哪些高阶用法？](#46.ControlNet有哪些高阶用法？)
- [47.ControlNet中Balanced、My prompt is more important、ControlNet is more important三种模式的区别是什么？](#47.ControlNet中Balanced、My-prompt-is-more-important、ControlNet-is-more-important三种模式的区别是什么？)
- [48.ControlNet 1.1与ControlNet相比，有哪些改进？](#48.ControlNet-1.1与ControlNet相比，有哪些改进？)
- [49.介绍一下ControlNet Canny条件控制的原理](#49.介绍一下ControlNet-Canny条件控制的原理)
- [50.介绍一下ControlNet Depth控制条件的原理](#50.介绍一下ControlNet-Depth控制条件的原理)
- [51.介绍一下ControlNet NormalMap控制条件的原理](#51.介绍一下ControlNet-NormalMap控制条件的原理)
- [52.介绍一下ControlNet OpenPose控制条件的原理](#52.介绍一下ControlNet-OpenPose控制条件的原理)
- [53.介绍一下ControlNet MLSD控制条件的原理](#53.介绍一下ControlNet-MLSD控制条件的原理)
- [54.介绍一下ControlNet Lineart控制条件的原理](#54.介绍一下ControlNet-Lineart控制条件的原理)
- [55.介绍一下ControlNet SoftEdge控制条件的原理](#55.介绍一下ControlNet-SoftEdge控制条件的原理)
- [56.介绍一下ControlNet Scribble/Sketch控制条件的原理](#56.介绍一下ControlNet-Scribble/Sketch控制条件的原理)
- [57.介绍一下ControlNet Segmentation控制条件的原理](#57.介绍一下ControlNet-Segmentation控制条件的原理)
- [58.介绍一下ControlNet Shuffle控制条件的原理](#58.介绍一下ControlNet-Shuffle控制条件的原理)
- [59.介绍一下ControlNet Tile/Blur控制条件的原理](#59.介绍一下ControlNet-Tile/Blur控制条件的原理)
- [60.介绍一下Controlnet Inpaint控制条件的原理](#60.介绍一下Controlnet-Inpaint控制条件的原理)
- [61.介绍一下ControlNet InstryctP2P控制条件的原理](#61.介绍一下ControlNet-InstryctP2P控制条件的原理)
- [62.介绍一下一下ControlNet Reference-only控制条件的原理](#62.介绍一下一下ControlNet-Reference-only控制条件的原理)
- [63.介绍一下ControlNet Recolor控制条件的原理](#63.介绍一下ControlNet-Recolor控制条件的原理)
- [64.训练ControlNet模型的流程中有哪些关键参数？](#64.训练ControlNet模型的流程中有哪些关键参数？)
- [65.ControlNet模型的训练流程一般包含哪几部分核心内容？](#65.ControlNet模型的训练流程一般包含哪几部分核心内容？)
- [66.ControlNet有哪些万金油级应用案例？](#66.ControlNet有哪些万金油级应用案例？)
- [67.SD/SDXL/FLUX.1 ControlNet之间有哪些区别？](#67.SD/SDXL/FLUX.1-ControlNet之间有哪些区别？)
- [68.介绍一下Controlnet-Union的原理](#68.介绍一下Controlnet-Union的原理)
- [69.ControlNet是如何起作用的？](#69.ControlNet是如何起作用的？)
- [70.ControlNet模型包含十几种控制功能可以归位几大类？](#70.ControlNet模型包含十几种控制功能可以归位几大类？)


<h2 id="2.ControlNet的模型结构与原理">2.ControlNet的模型结构与原理 </h2>

![](./imgs/Controlnet.png)

权重克隆：ControlNet 将大型扩散模型的权重克隆为两个副本，一个“可训练副本”和一个“锁定副本”。锁定副本保留了从大量图像中学习到的网络能力，而可训练副本则在特定任务的数据集上进行训练，以学习条件控制。

零卷积：ControlNet 引入了一种特殊类型的卷积层，称为“零卷积”。这是一个 1x1 的卷积层，其权值和偏差都初始化为零。零卷积层的权值会从零逐渐增长到优化参数，这样设计允许模型在训练过程中逐渐调整和学习条件控制，而不会对深度特征添加新的噪声。

特征融合：ControlNet 通过零卷积层将额外的条件信息融合到神经网络的深层特征中。这些条件可以是姿势、线条结构、颜色分布等，它们作为输入调节图像，引导图像生成过程。

灵活性和扩展性：ControlNet 允许用户根据需求选择不同的模型和预处理器进行组合使用，以实现更精准的图像控制和风格化。例如，可以结合线稿提取、颜色控制、背景替换等多种功能，创造出丰富的视觉效果。

**Controlnet如何处理条件图的？**

我们知道在 sd 中，模型会使用 VAE-encoder 将图像映射到隐空间，512×512 的像素空间图像转换为更小的 64×64 的潜在图像。而 controlnet 为了将条件图与 VAE 解码过的特征向量进行相加，controlnet 使用了一个小型的卷积网络，其中包括一些普通的卷积层，搭配着 ReLU 激活函数来完成降维的功能。


**加入Controlnet训练后，训练时间和显存的变化？**

在论文中，作者提到，与直接优化 sd 相比，优化 controlnet 只需要 23% 的显存，但是每一个 epoch 需要额外的 34% 的时间。可以方便理解的是，因为 controlnet 其实相当于只优化了unet-encoder，所以需要的显存较少，但是 controlnet 需要走两个网络，一个是原 sd 的 unet，另一个是复制的 unet-encoder，所以需要的时间会多一些。


<h2 id="27.CreatiLayout的框架和原理">27.CreatiLayout的框架和原理</h2>

论文链接：[[2412.03859\] CreatiLayout: Siamese Multimodal Diffusion Transformer for Creative Layout-to-Image Generation](https://arxiv.org/abs/2412.03859)

![image-20250113201544960](./imgs/siamLayout.png)

1. 整体架构

- 基于多模态扩散变换器(MM-DiT)设计
- 将布局作为与图像和文本同等重要的独立模态
- 采用孪生(Siamese)分支结构处理不同模态间的交互

2.主要创新点：SiamLayout架构 

这个架构通过两个关键设计解决了模态竞争问题：

a) 独立模态处理：

- 使用单独的transformer参数处理布局信息
- 使布局与图像和文本具有同等地位

b) 孪生分支结构：

- 将三模态交互解耦为两个平行分支:

  - 图像-文本分支
  - 图像-布局分支

- 后期再融合两个分支的输出

- 这种设计避免了模态间的直接竞争

- 作者还对比了其他两种架构变体：

  - Layout Adapter：通过cross-attention引入布局信息
  - M³-Attention：直接将三个模态放在一起做attention

  但实验表明SiamLayout的效果最好，主要原因是它避免了模态间的直接竞争，让每个模态都能充分发挥作用。

  示例：

![image-20250113201809122](./imgs/creatiLayout_示例.png)


<h2 id="28.Ctrl-X的框架和原理">28.Ctrl-X的框架和原理</h2>

论文链接：[2406.07540](https://arxiv.org/pdf/2406.07540

Ctrl-X 是一个训练无关、指导无关的框架，通过操控预训练的 T2I 扩散模型的特征层，提供对生成图像结构（Structure）和外观（Appearance）的控制。

![image-20250113203140236](./imgs/ctrl-x.png)

该框架 **Ctrl-X** 的核心流程可概括为以下几个关键步骤：

1. **结构与外观特征提取**：通过前向扩散过程分别对输入的结构图像 (x^s_t) 和外观图像 (x^a_t) 添加噪声后，将其输入到预训练的扩散模型中，提取卷积特征和自注意力特征。
2. **特征注入与迁移**：
   - **特征注入**：将结构图像的卷积特征和注意力特征注入到目标图像 (x^o_t) 的生成过程中，确保生成图像的结构与输入对齐。
   - **外观迁移**：利用输入外观图像和目标图像的自注意力对应关系，计算加权的均值 (M) 和标准差 (S)，用于对目标图像的特征进行归一化，实现空间感知的外观迁移。
3. **生成过程**：在每一步生成中，将上述注入的结构和外观特征结合，逐步生成符合目标结构与外观的图像。

**优势**：

- 无需额外训练，直接在预训练扩散模型上运行。
- 支持任意类型的结构和外观输入，具有高度的灵活性和高效性。

示例：

![image-20250113203422258](./imgs/ctrl-x-示例.png)


<h2 id="29.OMNIBOOTH的框架和原理">29.OMNIBOOTH的框架和原理</h2>

论文链接：[2410.04932](https://arxiv.org/pdf/2410.04932)

![image-20250113202957539](./imgs/OmniBooth.png)

核心架构分为三部分：

1.双路输入处理：

- 文本路径：Instance Prompt → Text Encoder → 文本嵌入
- 图像路径：Image references → DINO v2 → Spatial Warping → 图像嵌入

2.潜在控制信号(Latent Control Signal)：

- 维度为C×H'×W'的特征空间
- 通过Paint操作融合文本嵌入
- 通过Spatial Warping融合图像特征
- 作为统一的控制信号输入到生成网络

3.生成网络：

- Feature Alignment进行特征对齐
- Diffusion UNet生成最终图像
- 同时接收Global Prompt作为全局引导

示例：

![image-20250113203056301](./imgs/omnibooth_示例.png)


<h2 id="32.ReCo的框架和原理">32.ReCo的框架和原理</h2>

论文链接：[[2211.15518\] ReCo: Region-Controlled Text-to-Image Generation](https://arxiv.org/abs/2211.15518)

#### 框架架构

1. ReCo基于Stable Diffusion（SD）模型改进，主要包含以下模块：
   1. **双模态输入序列**：在传统文本令牌（Text Tokens）的基础上，引入**位置令牌（Position Tokens）**。这些位置标记通过坐标编码（如边界框的归一化坐标）生成，允许用户在输入查询中混合自由文本描述和区域位置信息。
      - 示例输入格式：
        `"a kitchen with <576> <553> <791> <979> stainless steel appliances and a counter"`
   2. **扩展的文本编码器**：沿用CLIP ViT-L/14文本编码器，但额外支持位置标记的嵌入，联合文本和位置标记生成条件嵌入向量。
   3. **扩散模型架构**：保持SD的U-Net架构，但通过微调使其能结合位置信息进行去噪。位置标记与文本标记通过交叉注意力机制共同指导图像生成。

![image-20250223194247270](./imgs/reco.png)

#### **关键原理**

1. **区域指令的可控性**：
    ReCo通过位置标记直接指定目标区域（如物体位置或范围），降低生成过程中的空间歧义。例如，用户可精确描述"沙发在画面左侧（区域<0, 0, 0.5, 1>）"，从而避免模型对布局的随机猜测。
2. **训练数据构建**：
   - 使用自动标注工具（如GIT captioning模型）为图像中的裁剪区域生成区域描述。
   - 对位置标记进行随机化裁剪与坐标编码，增强模型对多样化区域指令的泛化能力。
3. **保留预训练能力**：
    ReCo仅微调SD模型的文本编码器和交叉注意力层，最大限度保留原有生成质量，并适应区域控制任务。

下面是示例：

![image-20250223194333168](./imgs/reco_example.png)



<h2 id="33.Be-Yourself（Bounded-Attention-for-Multi-Subject-Text-to-Image-Generation）的框架和原理">33.Be Yourself（Bounded Attention for Multi-Subject Text-to-Image Generation）的框架和原理</h2>

论文链接：[[2403.16990\] Be Yourself: Bounded Attention for Multi-Subject Text-to-Image Generation](https://arxiv.org/abs/2403.16990)

#### **核心问题**

现有文本到图像扩散模型（如Stable Diffusion）在生成包含多个语义相似主体的复杂场景时，常出现**语义泄漏**（Semantic Leakage）问题，导致主体特征混淆（例如，“3只姜黄色小猫和2只灰色小猫”可能混合颜色或形态）。

![image-20250223200119401](./imgs/beyourself_example_for_issue.png)

论文提出**Bounded Attention**（有界注意力）机制，通过约束交叉注意力和自注意力层的相互作用，分割不同主体的生成流程，主要包含以下模块：

1. **Bounded Guidance**
   - **功能**：在去噪过程的早期阶段，通过用户提供的布局（如bounding boxes）生成粗略的语义分割掩模，引导各主体区域的空间定位。
   - **原理**：限制不同区域的注意力交互，抑制跨区域语义干扰（如避免“姜黄色小猫”的特征泄漏到“灰色小猫”区域）。
2. **Bounded Denoising**
   - **功能**：在去噪后期，细化各主体的细节特征，确保其与文本描述严格对齐（如颜色、纹理等）。
   - **原理**：通过修改自注意力层中的查询（Query）和键（Key）的相似性，强制不同主体的特征独立性。
3. **Mask Refinement**
   - **功能**：动态优化分割掩模，提升主体边界清晰度。
   - **原理**：结合去噪过程中的隐变量特征，迭代更新掩模，避免区域重叠导致的特征混合。

![image-20250223195927919](./imgs/beyouself.png)

下面是示例：

![image-20250223200443289](./imgs/BA_example.png)

<h2 id="34.IFadapter的框架和原理">34.IFadapter的框架和原理</h2>

### 研究背景

- 当前的文本到图像（Text-to-Image, T2I）扩散模型在生成单个实例的高质量图像方面表现出色，但在处理多个实例的精确位置和特征生成时存在局限性。
- 布局到图像（Layout-to-Image, L2I）任务通过引入边界框作为空间控制信号解决了位置问题，但在生成精确的实例特征方面仍有不足。

论文提出了一个更具挑战性的任务：**实例特征生成（Instance Feature Generation, IFG）**，旨在同时确保生成内容的位置准确性和特征忠实度。



### IFAdapter框架

IFAdapter（实例特征适配器）是为解决IFG任务而设计的模型，主要包含两个核心组件：

#### 1. 外观令牌（Appearance Tokens）

- 解决问题：现有模型主要使用单一上下文化令牌（EoT令牌）来指导实例特征生成，无法捕捉高频细节特征
- 原理：引入可学习的外观查询，从实例描述中提取特定特征信息，形成外观令牌，与EoT令牌一起工作
- 优势：能够更精确地控制实例特征的生成，特别是复杂的纹理、混合颜色等细节

#### 2. 实例语义图（Instance Semantic Map）

- 解决问题：现有序列到2D定位条件无法提供足够强的空间先验
- 原理：构建2D语义图将实例特征与指定空间位置关联起来，提供增强的空间引导
- 特点：在多个实例重叠的区域，采用门控语义融合机制解决特征混淆问题
- 实现：仅在扩散模型的部分交叉注意力层中集成语义图，实现松散耦合

<img src="./imgs/ifadapter.png" alt="image-20250309161515489" style="zoom:67%;" />

效果如下：

![image-20250309161925832](./imgs/ifadapter_example.png)

IFAdapter的即插即用设计使其能够无缝赋能各种社区模型，应用于图形设计和艺术设计等需要局部高级细节的场景。

这项研究为解决文本到图像生成中的精细控制问题提供了一种有效的方法，在保持位置准确性的同时提高了特征表现力，推动了可控图像生成技术的发展。

<h2 id="35.LAW-Diffusion的框架和原理">35.LAW-Diffusion的框架和原理</h2>

LAW-Diffusion是一种语义可控的布局感知扩散模型，其核心思想是解析对象之间的空间依赖关系，生成具有协调一致对象关系的复杂场景图像。该框架主要包含以下组件：

### 1. 空间依赖关系解析器（Spatial Dependency Parser）

不同于之前仅探索类别感知关系的方法，LAW-Diffusion引入了空间依赖关系解析器，用于编码对象之间的位置感知语义连贯性：

- **对象区域图（Object Region Maps）**：为每个对象实例化区域语义表示，将类别嵌入填充到其边界框指定的区域
- 位置感知跨对象注意力（Location-aware Cross-object Attention）：
  - 将对象区域图分割成区域片段
  - 对相同位置的区域片段执行多头注意力操作
  - 使用可学习的聚合令牌捕获位置感知的组合语义
  - 重新组合聚合片段得到布局嵌入（Layout Embedding）

这种方式同时捕获了类别感知和位置感知的依赖关系，确保在生成图像的局部片段时，能够准确指定对象在特定位置的可能重叠情况。

![image-20250309164931421](./imgs/Spatial Dependency Parser.png)

### 2. 自适应引导调度（Adaptive Guidance Schedule）

为了平衡区域语义对齐与对象纹理保真度之间的权衡，LAW-Diffusion提出了自适应引导调度策略：

- 在采样阶段使用余弦形式的引导幅度衰减函数
- 从初始较大的引导比例逐渐衰减到较小值
- 早期阶段强调语义控制，后期阶段注重纹理细节

这种策略类似于人类绘图时先构思整体语义，再细化细节的直觉过程。

![image-20250309165104833](./imgs/Adaptive Guidance Schedule.png)

### 3. 布局感知潜在嫁接（Layout-aware Latent Grafting）

LAW-Diffusion还支持实例级别的重构能力，包括添加/移除/重新设计生成场景中的实例：

- 从已生成图像的扩散潜在表示中，提取边界框外的区域
- 在相同噪声级别下，将该区域嫁接到由新布局引导的目标潜在表示上
- 通过交替重组局部区域语义和去噪这些嫁接的潜在表示，实现实例重构

![image-20250309165119951](./imgs/Layout-aware Latent Grafting.png)

示例如下：

![image-20250309165333162](./imgs/LAW-Diffusion.png)

LAW-Diffusion通过引入空间依赖关系解析、自适应引导调度和布局感知潜在嫁接等创新技术，显著提升了布局到图像生成的效果，特别是在保持复杂场景中对象之间合理和协调的关系方面。该方法为控制复杂场景生成提供了新的思路，具有重要的理论和应用价值。

<h2 id="36.Check,Locate,Rectify（A-Training-Free-Layout-Calibration-System-for-Text-to-Image-Generation）的框架和原理">36.Check, Locate, Rectify（A Training-Free Layout Calibration System for Text-to-Image Generation）的框架和原理</h2>

本研究论文介绍了SimM，这是一种新颖的系统，旨在解决文本到图像生成中的一个常见挑战：准确实现文本提示中的空间布局指令。

### SimM方法

SimM采用“检查-定位-纠正”流程，无需额外训练即可干预生成过程：

1. **检查**：
   - 使用预定义词汇确定提示是否包含布局要求
   - 使用依赖解析和启发式规则生成对象的目标布局
   - 评估当前生成是否可能偏离这些要求
2. **定位**：
   - 在早期去噪步骤中识别对象当前被放置的位置
   - 使用注意力图找到每个对象的激活区域
3. **纠正**：
   - 将激活从错误位置转移到目标位置
   - 增强目标区域的激活并抑制其他区域的激活
   - 防止不同对象之间的激活重叠

![image-20250309163901844](./imgs/SimM.png)



示例如下：

![image-20250309163945734](./imgs/SimM_example.png)

### 关键创新

1. **无需训练的实现**：可与现有的预训练模型配合使用，无需微调
2. **处理两种空间关系**:
   - 相对关系（例如，“一只狗在猫的左边”）
   - 最高级关系（例如，“左边的一朵花”）
3. **高效的布局生成**：自动从文本中推导出目标布局，无需手动输入
4. **最小的计算开销**：直接修改注意力图，无需复杂的优化

该技术可以改善创意应用中的用户控制，允许在文本到图像生成中更精确地指定布局，而无需具备布局技术知识或额外训练。

<h2 id="37.Conceptrol的框架和原理">37.Conceptrol的框架和原理</h2>

个性化图像生成面临着一个根本性的困境：如何在遵循创意文本提示的同时，平衡从参考图像中保留对象身份。目前的方法主要分为两类：

1. **微调方法**，如 DreamBooth 和 Textual Inversion，每种新概念都需要大量的计算资源和训练时间。
2. **零样本适配器**，如 IP-Adapter 和 OmniControl，效率更高，但在遵循复杂的提示时，通常难以维持概念身份。

核心挑战是，现有的零样本适配器无法有效地利用扩散模型的文本理解能力。它们分别处理参考图像和文本提示，错过了它们之间的关键联系。
Conceptrol 引入了一种非常简单但有效的方法来解决这个困境。关键的见解是，扩散模型已经包含强大的“文本概念掩码”，可以引导注意力集中到图像的相关部分。

![image-20250420205157570](./imgs/conceptrol.png)

该方法通过一个三步过程工作：

1. **文本概念识别**：系统首先识别提示中与参考图像相对应的文本概念（例如，雕像图像的“雕像”）。
2. **概念掩码提取**：它从模型的内部表示中提取与该文本概念相对应的注意力掩码。
3. **引导注意力**：该掩码用于约束参考图像的视觉信息的应用位置，确保它与文本提示的意图对齐。

这种方法不需要额外的训练，并且可以作为现有零样本适配器的即插即用增强功能。它可以应用于基于 U-Net 的模型（如 Stable Diffusion）和基于 DiT 的模型（如 FLUX）。


<h2 id="39.Prompt-to-Prompt是什么方法？">39.Prompt-to-Prompt是什么方法？</h2>

### 1. 方法概述

**Prompt-to-Prompt (P2P)**是一种基于文本的图像编辑方法，通过操控跨注意力机制，实现仅通过文本提示即可进行精细化图像编辑，而无需额外的用户输入（如遮罩或手动编辑）。核心思想在于利用扩散模型中的**跨注意力层**，操控像素与文本标记之间的交互关系，从而在生成过程中保留原始图像的结构和布局。![image-20241021170146517](./imgs/P2P.png)

### 2. 方法细节

1. **跨注意力机制的作用**:
    - 在图像生成过程中，扩散模型通过跨注意力层将文本嵌入和视觉特征融合，每个文本标记会生成对应的空间注意力图，决定了文本中每个词汇对图像不同区域的影响。
    - 通过控制这些跨注意力图，研究人员能够保留图像的原始结构，同时在不同的生成步骤中调整文本对生成结果的影响。

2. **编辑策略**:
    - **单词替换**: 通过将跨注意力图从原始提示转移到新的文本提示，方法能够在替换部分内容（如“狗”替换为“猫”）的同时保持场景的整体布局。
    - **添加新短语**: 当用户在原始提示上增加描述（如增加风格或颜色），方法会将未改变的部分的注意力图保持一致，使新元素自然融入图像。
    - **调整单词权重**: 方法允许调整某个词的影响程度，实现类似“滑块控制”的效果，使得用户可以增强或减弱某些特定词汇对图像生成的作用。

3. **编辑流程**:
    - 编辑的核心步骤是通过注入原始图像的跨注意力图，将其与新提示中的注意力图结合，并在扩散过程的不同阶段应用这些调整。
    - 通过**时间戳参数**，方法还能调节注意力图的影响范围，从而控制生成图像的保真度和平滑度。

### 3. 应用示例

1. **局部编辑**:
    - 通过调整文本提示中的单词，可以局部替换图像中的特定对象，如将“柠檬蛋糕”变成“南瓜蛋糕”。
    - 这种方法无需用户提供遮罩，能够自然地改变图像中的纹理和物体形状。

2. **全局编辑**:
    - 添加新描述词语使得可以实现全局风格转换或环境变化，例如为图像添加“雪”或改变光照效果。
    - 方法能够保留图像的整体构图，确保新的风格或背景不会破坏原有的视觉结构。

3. **风格化**:
    - 通过在提示中添加风格描述，方法可以将草图转换为照片真实感图像，或生成各种艺术风格的图像。

### 4. 方法优势

- **仅需文本控制**: 不依赖用户手动输入的遮罩或结构化标记，仅通过修改提示文本即可实现多样化和精细化的图像编辑。
- **高保真度**: 方法能够在保持原始图像结构和布局的同时，准确生成与修改提示相符的图像。
- **实时性**: 相比于传统的训练或微调模型，这种基于扩散模型内部跨注意力的操控方法不需要额外的数据或优化步骤。

本文的方法展示了通过文本操控生成模型内部机制来实现图像编辑的新可能性，为未来更加智能、直观的图像生成和编辑工具奠定了基础。


<h2 id="40.InstructPix2Pix的训练和推理流程是什么样的？">40.InstructPix2Pix的训练和推理流程是什么样的？</h2>

论文链接：[2211.09800](https://arxiv.org/pdf/2211.09800)

![image-20241021152129263](./imgs/ip2p.png)

### 1. 训练流程

1. **生成训练数据**：
   - 使用 **GPT-3** 生成文本三元组，包括输入图像描述、编辑指令、编辑后的图像描述。
   - 利用 **Stable Diffusion** 和 **Prompt-to-Prompt** 方法，根据文本生成配对的图像（编辑前和编辑后），并通过 **CLIP** 过滤确保图像质量和一致性。
2. **训练 InstructPix2Pix 模型**：
   - 使用 **Stable Diffusion** 的预训练权重进行初始化。
   - 输入原始图像、编辑指令和目标编辑后的图像。
   - 训练目标是最小化潜在扩散目标函数，应用无分类器引导技术以平衡图像和文本指令的影响。

### 2. 推理流程

1. **输入**：
   - 一张待编辑的真实图像和一条人类编写的编辑指令。

2. **处理**：
   - 将输入图像编码到潜在空间。
   - 应用条件扩散模型，根据输入图像和文本指令生成编辑后的潜在表示。
   - 使用无分类器引导，通过调整两个引导尺度（s_I 和 s_T）平衡图像和指令的影响。

3. **输出**：
   - 将生成的潜在表示解码为编辑后的图像，通常生成 **512x512** 分辨率的结果。
   - 每张图像的编辑过程在 **A100 GPU** 上大约需要 **9 秒**，使用 **100** 个去噪步骤。


<h2 id="41.ControlNet系列模型有多少种控制条件？">41.ControlNet系列模型有多少种控制条件？</h2>


<h2 id="42.ControlNet的最小单元是什么样的？">42.ControlNet的最小单元是什么样的？</h2>

下图是ControlNet模型的最小单元：

![ControlNet模型的最小单元结构示意图](./imgs/ControlNet模型的最小单元结构示意图.png)

从上图可以看到，**在使用ControlNet模型之后，Stable Diffusion/FLUX.1模型的权重被复制出两个相同的部分，分别是“锁定”副本（locked）权重和“可训练”副本（trainable copy）权重**。

**我们如何理解这两个副本权重呢？** Rocky从训练角度和推理角度给大家进行通俗易懂的讲解。

**首先不管是训练阶段还是推理阶段，ControlNet都在“可训练”副本上输入控制条件** $c$，然后将“可训练”副本输出结果和原来Stable Diffusion/FLUX.1模型的“锁定”副本输出结果**相加（add）**获得最终的输出结果。

在训练阶段，**其中“锁定”副本中冻结参数，权重保持不变，保留了Stable Diffusion/FLUX.1模型原本的能力**；与此同时，**使用新数据对“可训练”副本进行微调训练，学习数据中的控制条件信息**。因为有Stable Diffusion/FLUX.1模型作为预训练权重，**复制“可训练”副本而不是直接训练原始权重还能避免数据集较小时的过拟合**，所以我们使用常规规模数据集（几K-几M级别）就能对控制条件进行学习训练，同时不会破坏Stable Diffusion/FLUX.1模型原本的能力（从数十亿张图像中学习到的大型模型的能力）。

另外，大家可能发现了**ControlNet模型的最小单元结构中有两个zero convolution模块，它们是1×1卷积，并且在微调训练时权重和偏置都初始化为零（zero初始化）**。这样一来，在我们开始训练ControlNet之前，所有zero convolution模块的输出都为零，使得ControlNet完完全全就在原有Stable Diffusion/FLUX.1底模型的能力上进行微调训练，这样可以尽量避免训练加入的初始噪声对ControlNet“可训练”副本权重的破坏，保证了不会产生大的能力偏差。

<h2 id="43.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？">43.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？</h2>

大家很可能就会有一个疑问，如果zero convolution模块的初始权重为零，那么梯度也为零，ControlNet模型将不会学到任何东西。**那么为什么“zero convolution模块”有效呢？（AIGC算法面试必考点）**

Rocky进行下面的推导，相信大家对一切都会非常清晰明了：

我们可以假设ControlNet的初始权重为： $y=wx+b$ ，然后我们就可以得到对应的梯度求导：

$$\frac{\partial y}{\partial w}=x,\frac{\partial y}{\partial x}=w,\frac{\partial y}{\partial b}=1$$

如果此时 $w=0$ 并且 $x \neq 0$ ，然后我们就可以得到：

$$\frac{\partial y}{\partial w} \neq 0,\frac{\partial y}{\partial x}=0,\frac{\partial y}{\partial b}\neq 0$$

这就意味着只要 $x \neq 0$ ，一次梯度下降迭代将使w变成非零值。然后就得到： $\frac{\partial y}{\partial x}\neq 0$ 。**这样就能让zero convolution模块逐渐成为具有非零权重的卷积层，并不断优化参数权重**。

<h2 id="44.ConrtolNet是如何训练的？">44.ConrtolNet是如何训练的？</h2>

我们对ControlNet整体训练过程进行拆解理解。在我们不使用ControlNet模型时，**可以将Stable Diffusion/FLUX.1底模型的图像生成过程表达为：**

![不使用ControlNet模型时扩散模型推理示意图](./imgs/不使用ControlNet模型时扩散模型推理示意图.png)

接着，我们在此基础上假设将训练的所有参数锁定在 $\Theta$ 中，然后将其复制为可训练的副本 $\Theta_{c}$ 。复制的 $\Theta_{c}$ 使用额外控制条件信息c进行训练。因此在使用ControlNet之后，**Stable Diffusion/FLUX.1底模型 + ControlNet模型整体的图像生成表达式转化成为：**

![StableDiffusion和FLUX.1底模型+ControlNet模型的整体图像生成过程](./imgs/StableDiffusion和FLUX.1底模型+ControlNet模型的整体图像生成过程.png)

其中 $Z = F(c; \Theta)$ 代表了zero convolution模块， $\Theta_{z1}$ 和 $\Theta_{z2}$ 代表了前后两个zero convolution层的参数权重， $\Theta_{c}$ 则代表了ControlNet的参数权重。

由于训练开始前zero convolution模块的输出都为零，所以ControlNet未经训练时的初始输出为0：

$$\begin{cases} 
\mathcal{Z}\left(\boldsymbol{c};\Theta_{z1}\right) = 0 \\ 
\mathcal{F}\left(x + \mathcal{Z}\left(\boldsymbol{c};\Theta_{z1}\right);\Theta_{\mathrm{c}}\right) = \mathcal{F}\left(x;\Theta_{\mathrm{c}}\right) = \mathcal{F}(x;\Theta) \\
\mathcal{Z}\left(\mathcal{F}\left(x + \mathcal{Z}\left(\boldsymbol{c};\Theta_{z1}\right);\Theta_{\mathrm{c}}\right);\Theta_{z2}\right) = \mathcal{Z}\left(\mathcal{F}\left(x;\Theta_{\mathrm{c}}\right);\Theta_{z2}\right) = \mathbf{0} 
\end{cases}$$

由此可知，**在ControlNet微调训练初始阶段对Stable Diffusion/FLUX.1底模型权重是没有任何影响的，能让底模型原本的性能完整保存**，之后ControlNet的训练也只是在原Stable Diffusion/FLUX.1底模型基础上进行优化。

总的来说，**ControlNet的本质原理使得训练后的模型鲁棒性好，能够避免模型过拟合，并在特定条件场景下具有良好的泛化性，同时能够在小规模数据和消费级显卡上进行训练**。

<h2 id="45.ControlNet的损失函数是什么？">45.ControlNet的损失函数是什么？</h2>


<h2 id="46.ControlNet有哪些高阶用法？">46.ControlNet有哪些高阶用法？</h2>


<h2 id="47.ControlNet中Balanced、My-prompt-is-more-important、ControlNet-is-more-important三种模式的区别是什么？">47.ControlNet中Balanced、My prompt is more important、ControlNet is more important三种模式的区别是什么？</h2>


<h2 id="48.ControlNet-1.1与ControlNet相比，有哪些改进？">48.ControlNet 1.1与ControlNet相比，有哪些改进？</h2>

**ControlNet 1.1与ControlNet 1.0具有完全相同的模型架构。ControlNet 1.1主要是在ControlNet 1.0的基础上进行了优化训练，提高了鲁棒性和控制效果，同时发布了几个新的ControlNet模型。**

从ControlNet 1.1开始，ControlNet模型将使用标准的命名规则（SCNNR）来命名所有模型，这样我们在使用时也能更加方便与清晰。具体的命名规则如下图所示：

![ControlNet1.1模型命名规则.png](./imgs/ControlNet1.1模型命名规则.png)

ControlNet 1.1一共发布了14个模型（11个成品模型和3 个实验模型）：

```bash
control_v11p_sd15_canny
control_v11p_sd15_mlsd
control_v11f1p_sd15_depth
control_v11p_sd15_normalbae
control_v11p_sd15_seg
control_v11p_sd15_inpaint
control_v11p_sd15_lineart
control_v11p_sd15s2_lineart_anime
control_v11p_sd15_openpose
control_v11p_sd15_scribble
control_v11p_sd15_softedge
control_v11e_sd15_shuffle（实验模型）
control_v11e_sd15_ip2p（实验模型）
control_v11f1e_sd15_tile（实验模型）
```

<h2 id="49.介绍一下ControlNet-Canny条件控制的原理">49.介绍一下ControlNet Canny条件控制的原理</h2>


<h2 id="50.介绍一下ControlNet-Depth控制条件的原理">50.介绍一下ControlNet Depth控制条件的原理</h2>


<h2 id="51.介绍一下ControlNet-NormalMap控制条件的原理">51.介绍一下ControlNet NormalMap控制条件的原理</h2>


<h2 id="52.介绍一下ControlNet-OpenPose控制条件的原理">52.介绍一下ControlNet OpenPose控制条件的原理</h2>


<h2 id="53.介绍一下ControlNet-MLSD控制条件的原理">53.介绍一下ControlNet MLSD控制条件的原理</h2>


<h2 id="54.介绍一下ControlNet-Lineart控制条件的原理">54.介绍一下ControlNet Lineart控制条件的原理</h2>


<h2 id="55.介绍一下ControlNet-SoftEdge控制条件的原理">55.介绍一下ControlNet SoftEdge控制条件的原理</h2>


<h2 id="56.介绍一下ControlNet-Scribble/Sketch控制条件的原理">56.介绍一下ControlNet Scribble/Sketch控制条件的原理</h2>


<h2 id="57.介绍一下ControlNet-Segmentation控制条件的原理">57.介绍一下ControlNet Segmentation控制条件的原理</h2>


<h2 id="58.介绍一下ControlNet-Shuffle控制条件的原理">58.介绍一下ControlNet Shuffle控制条件的原理</h2>


<h2 id="59.介绍一下ControlNet-Tile/Blur控制条件的原理">59.介绍一下ControlNet Tile/Blur控制条件的原理</h2>


<h2 id="60.介绍一下Controlnet-Inpaint控制条件的原理">60.介绍一下Controlnet Inpaint控制条件的原理</h2>


<h2 id="61.介绍一下ControlNet-InstryctP2P控制条件的原理">61.介绍一下ControlNet InstryctP2P控制条件的原理</h2>


<h2 id="62.介绍一下一下ControlNet-Reference-only控制条件的原理">62.介绍一下一下ControlNet Reference-only控制条件的原理</h2>


<h2 id="63.介绍一下ControlNet-Recolor控制条件的原理">63.介绍一下ControlNet Recolor控制条件的原理</h2>


<h2 id="64.训练ControlNet模型的流程中有哪些关键参数？">64.训练ControlNet模型的流程中有哪些关键参数？</h2>


<h2 id="65.ControlNet模型的训练流程一般包含哪几部分核心内容？">65.ControlNet模型的训练流程一般包含哪几部分核心内容？</h2>

ControlNet系列模型的训练流程主要分成以下几个步骤：

1. 设计我们想要的额外控制条件：除了上面章节中讲到的控制条件，我们还可以根据实际需求自定义一些控制条件，从而使用ControlNet控制Stable Diffusion/FLUX.1朝着我们想要的细粒度方向生成内容。
2. 构建训练数据集：确定好额外控制条件后，我们就可以开始构建训练数据集了。ControlNet数据集中需要包含三个维度的信息：Ground Truth图片、作为控制条件（Conditional）的图片，以及对应的Caption标签。
3. 训练我们自己的ControlNet模型：训练数据集构建好后，我们就可以开始训练自己的ControlNet模型了，我们需要一个至少8G显存的GPU才能满足ControlNet模型的训练要求。

<h2 id="66.ControlNet有哪些万金油级应用案例？">66.ControlNet有哪些万金油级应用案例？</h2>


<h2 id="67.SD/SDXL/FLUX.1-ControlNet之间有哪些区别？">67.SD/SDXL/FLUX.1 ControlNet之间有哪些区别？</h2>


<h2 id="68.介绍一下Controlnet-Union的原理">68.介绍一下Controlnet-Union的原理</h2>

controlnet-union-sdxl-1.0模型的结构如下所示：

![controlnet-union-sdxl-1.0模型结构示意图](./imgs/controlnet-union-sdxl-1.0模型结构示意图.png)

Controlnet-union-sdxl-1.0模型模型的优化点：

1. **采用分桶训练技术**：对不同分辨率数据采用分桶训练策略，这样在推理时能够生成任意宽高比的高分辨率图像。
2. **海量高质量训练数据**：使用超过10M张高质量图像，数据集覆盖多样化的场景和内容。
3. **采用重新标注提示词**：使用CogVLM模型生成详细的Caption描述作为训练标签，使得模型具备优秀的提示词遵循能力。
4. **集成多种训练技巧**：包括但不限于数据增强、多目标损失函数、多分辨率训练等。
5. **参数效率高**：与原始ControlNet相比，参数量几乎未增加，网络参数和计算量无明显上升。
6. **支持多种控制条件**：兼容12种控制方式+5种高级编辑功能，每种条件的控制效果均不逊色于独立训练的ControlNet模型。
7. **支持多条件融合生成**：支持在推理时同时使用多条件的控制能力，多条件融合机制在训练中学习得到，无需手动设置超参数或设计提示词。
8. **兼容性强**：可与业界主流的SDXL模型、LoRA模型兼容使用。

controlnet-union-sdxl-1.0模型基于原始ControlNet架构，同时提出了**两个新模块Condition Transformer（条件变换器）和Control Encoder（控制编码器）**。

**在控制编码器中**，每个控制条件都被赋予一个特定的**控制类型标识符**。例如，OpenPose 对应标识符 (1, 0, 0, 0, 0, 0)，深度图对应 (0, 1, 0, 0, 0, 0)。**当存在多个条件时，例如同时使用 OpenPose 和深度图，其标识符将合并为 (1, 1, 0, 0, 0, 0)**。在控制编码器中，这些标识符通过正弦位置编码转换为嵌入向量，随后通过线性层将其投影至与时间嵌入相同的维度。**控制类型特征会与时间嵌入相加，从而在网络中传递不同控制类型的全局信息**。这一简洁设计有助于ControlNet区分各类控制条件，因为时间嵌入通常对整体网络具有广泛影响。无论是单一条件还是多条件组合，均对应唯一的控制类型标识符。

实际的控制类型标识符如下所示：

0 -- openpose
1 -- depth
2 -- thick line(scribble/hed/softedge/ted-512)
3 -- thin line(canny/mlsd/lineart/animelineart/ted-1280)
4 -- normal
5 -- segment

**在条件变换器中**，对ControlNet进行了扩展，使其能够同时处理多个控制输入。条件变换器的作用在于整合不同图像条件的特征。controlnet-union-sdxl-1.0模型的两大创新点在于：

1. 首先，不同条件共享同一条件控制编码器，从而使网络结构更为简洁与轻量。
2. 其次，引入了一个Transformer层，用于在原始图像特征与条件图像特征之间交换信息。同时并未直接采用Transformer的输出，而是利用其预测原始条件特征的调整量（即条件偏差）。**这种设计类似于 ResNet 的残差思想，实验表明该结构能显著提升模型性能**。

**同时针对多条件的同时控制，对ControlNet的条件编码器（Condition Encoder）也做了改进**。ControlNet原有的条件编码器由多个卷积层与Silu激活函数堆叠而成。**在保持其架构不变的基础上，controlnet-union-sdxl-1.0增加了卷积通道数，构建了一个更“宽”的编码器，这一改进显著提升了网络的表现能力**。原因在于，所有图像条件共享同一编码器，因此需要编码器具备更强的特征表示能力。原有结构对于单一条件可能足够，但在处理十余种条件时则显得力不从心。

**为了让controlnet-union-sdxl-1.0模型能够同时进行多条件的控制生成，官方设置了统一训练策略（Unified Training Strategy）进行多条件训练**。多条件训练有助于促进不同条件之间的融合，并增强模型的鲁棒性（因为单一条件所涵盖的知识有限）。


<h2 id="69.ControlNet是如何起作用的？">69.ControlNet是如何起作用的？</h2>

在以Stable Diffusion和FLUX.1为核心的AIGC图像生成过程中，想要ControlNet起作用，首先我们需要输入一张参考图，通过**预处理器** (Preprocessor)对输入参考图按一定的模式进行预处理，通常是使用传统的计算机视觉算法（如边缘检测、人体姿态估计、深度估计等）来从输入参考图中提取出纯粹的控制信息，也就是我们常说的**条件图像**(Conditioning Image)。

当然的，我们也可以不使用预处理功能，直接输入一张自己处理好的图片当作预处理图。下面是Rocky构建的ControlNet的条件图像处理流程图示，让大家能够更好的理解：

![ControlNet的条件图像处理流程](./imgs/ControlNet的条件图像处理流程.png)

接着条件图像信息通过ControlNet再注入到Stable Diffusion和FLUX.1中，再加上原本就直接注入到Stable Diffusion和FLUX.1中的文本信息和图像信息（可选，进行图生图任务），综合作用进行扩散过程，最终生成受条件信息控制的图像。

总的来说，ControlNet做的就是这样一件事：**它为扩散模型（如 Stable Diffusion/FLUX.1）提供一种额外的“约束”条件，引导AIGC大模型按照我们期望的构图、姿态或结构来生成图像，减少图像生成的随机**性。

为了大家方便的理解，**Rocky也制作了ControlNet推理的完整流程图**，大家可以直观的学习理解：

![完整的ControlNet模型推理流程](./imgs/完整的ControlNet模型推理流程.png)


<h2 id="70.ControlNet模型包含十几种控制功能可以归位几大类？">70.ControlNet模型包含十几种控制功能可以归位几大类？</h2>

Rocky认为这是一个非常重要的问题，ControlNet模型的各种控制功能非常多，我们需要进行归纳总结，才能更好的在AIGC时代中运用这些技术工具。

我们可以根据其处理信息的**类型**和**应用场景**归为几大类：

### 第一类：边缘与线条类
这类模型通过提取图像中的线条、轮廓或边缘信息来控制图像的结构和形状。它们通常用于精确的形状控制和线稿上色。

*   **Canny**：基于Canny边缘检测算法，提取图像中所有显著的边缘。
*   **MLSD**：专门用于检测建筑和室内设计中的**直线**，非常适合生成建筑草图、室内布局。
*   **Scribble**：将输入视为**涂鸦**或手绘草图，能够将非常粗略的线条转化为精致的图像。
*   **Soft Edge**：类似于Canny，但边缘更柔和、更粗，对自然图像（如动物、植物）的兼容性更好。
*   **Lineart**：专门用于提取**线稿**，尤其是从真实照片或艺术作品中提取，线条质量通常比Canny更高、更干净。

### 第二类：几何与3D信息类
这类模型使用从图像中推断出的3D信息（如深度、法线）来指导生成过程，从而控制物体的空间关系和立体感。

*   **Depth**：提取图像的**深度信息**，生成带有前景、中景、背景层次的图像，非常适合保持场景的立体感和空间一致性。
*   **Normal**：提取物体的**表面法线贴图**，它包含了物体表面细微的朝向信息，能生成光照和表面质感非常逼真的图像。

### 第三类：语义与内容信息类
这类模型使用更高层次的、经过抽象和理解的信息来控制生成内容，例如人体姿态、物体分割区域等。

*   **OpenPose**：检测图像中人物的**骨骼关键点**（姿势），可以精确控制生成人物的动作、姿态，甚至是手部动作和多人场景。
*   **Segmentation**：使用**语义分割图**，为图像中的不同部分（如天空、树木、人物、衣服）分配不同的颜色标签，从而对画面的每个区域进行像素级的精确控制。

### 第四类：风格与抽象信息类
这类模型不关注具体的形状或结构，而是关注图像的整体风格、颜色分布或纹理。

*   **Shuffle**：提取输入图像的**颜色分布和风格纹理**，将其应用到生成的新图像上，本质上是一种内容感知的风格迁移。
*   **Instruct Pix2Pix**：这是一个特殊模型，它不依赖于额外的控制图，而是直接接受**文字指令**来编辑图像。

### 第五类：特殊应用与重绘类
这类模型用于解决特定的图像生成或编辑任务。

*   **Tile**：用于**图像超分辨率**和**细节重绘**。它通过忽略输入图像的宏观结构，专注于局部纹理和细节，来对图像进行“放大并增强细节”的操作。
*   **Inpaint**：专门用于**局部重绘**。需要与Stable Diffusion的inpaint功能结合使用，通过提供蒙版区域和ControlNet的引导，在特定区域内进行高质量、与周围环境协调的重绘。

### 总结

| 类别 | 核心功能 | 包含模型 |
| :--- | :--- | :--- |
| **边缘与线条** | 控制形状、轮廓、结构 | Canny, MLSD, Scribble, Soft Edge, Lineart |
| **几何与3D** | 控制空间深度、立体感、表面朝向 | Depth, Normal |
| **语义与内容** | 控制人物姿态、物体分区 | OpenPose, Segmentation |
| **风格与抽象** | 控制颜色风格、纹理、根据指令编辑 | Shuffle, Instruct Pix2Pix |
| **特殊应用** | 图像放大、细节增强、局部重绘 | Tile, Inpaint |

这个分类方式可以帮助您更好地理解每个ControlNet模型的适用场景，并根据我们的创作需求快速选择合适的模型。在实际使用中，我们经常会将多个ControlNet组合使用，以达到更复杂和精确的控制效果。

---
