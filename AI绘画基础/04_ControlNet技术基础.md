# 目录

- [1.介绍一下ControlNet的模型结构与原理](#1.介绍一下ControlNet的模型结构与原理)
- [2.ControlNet的最小单元是什么样的？](#2.ControlNet的最小单元是什么样的？)
- [3.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？](#3.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？)
- [4.ControlNet系列模型有多少种控制条件？介绍一下各控制条件的原理与功能](#4.ControlNet系列模型有多少种控制条件？介绍一下各控制条件的原理与功能)
- [5.ConrtolNet是如何训练的？训练ControlNet模型的流程中有哪些关键参数？](#5.ConrtolNet是如何训练的？训练ControlNet模型的流程中有哪些关键参数？)
- [6.ControlNet的损失函数是什么？](#6.ControlNet的损失函数是什么？)
- [7.ControlNet中Balanced、My prompt is more important、ControlNet is more important三种模式的区别是什么？](#7.ControlNet中Balanced、My-prompt-is-more-important、ControlNet-is-more-important三种模式的区别是什么？)
- [8.ControlNet 1.1与ControlNet相比，有哪些改进？](#8.ControlNet-1.1与ControlNet相比，有哪些改进？)
- [9.介绍一下Controlnet-Union的原理](#9.介绍一下Controlnet-Union的原理)
- [10.ControlNet有哪些万金油级应用案例？](#10.ControlNet有哪些万金油级应用案例？)


<h2 id="1.介绍一下ControlNet的模型结构与原理">1.介绍一下ControlNet的模型结构与原理 </h2>

![](./imgs/Controlnet.png)

权重克隆：ControlNet 将大型扩散模型的权重克隆为两个副本，一个“可训练副本”和一个“锁定副本”。锁定副本保留了从大量图像中学习到的网络能力，而可训练副本则在特定任务的数据集上进行训练，以学习条件控制。

零卷积：ControlNet 引入了一种特殊类型的卷积层，称为“零卷积”。这是一个 1x1 的卷积层，其权值和偏差都初始化为零。零卷积层的权值会从零逐渐增长到优化参数，这样设计允许模型在训练过程中逐渐调整和学习条件控制，而不会对深度特征添加新的噪声。

特征融合：ControlNet 通过零卷积层将额外的条件信息融合到神经网络的深层特征中。这些条件可以是姿势、线条结构、颜色分布等，它们作为输入调节图像，引导图像生成过程。

灵活性和扩展性：ControlNet 允许用户根据需求选择不同的模型和预处理器进行组合使用，以实现更精准的图像控制和风格化。例如，可以结合线稿提取、颜色控制、背景替换等多种功能，创造出丰富的视觉效果。

**Controlnet如何处理条件图的？**

我们知道在 sd 中，模型会使用 VAE-encoder 将图像映射到隐空间，512×512 的像素空间图像转换为更小的 64×64 的潜在图像。而 controlnet 为了将条件图与 VAE 解码过的特征向量进行相加，controlnet 使用了一个小型的卷积网络，其中包括一些普通的卷积层，搭配着 ReLU 激活函数来完成降维的功能。


**加入Controlnet训练后，训练时间和显存的变化？**

在论文中，作者提到，与直接优化 sd 相比，优化 controlnet 只需要 23% 的显存，但是每一个 epoch 需要额外的 34% 的时间。可以方便理解的是，因为 controlnet 其实相当于只优化了unet-encoder，所以需要的显存较少，但是 controlnet 需要走两个网络，一个是原 sd 的 unet，另一个是复制的 unet-encoder，所以需要的时间会多一些。

**ControlNet是如何起作用的？**

在以Stable Diffusion和FLUX.1为核心的AIGC图像生成过程中，想要ControlNet起作用，首先我们需要输入一张参考图，通过**预处理器** (Preprocessor)对输入参考图按一定的模式进行预处理，通常是使用传统的计算机视觉算法（如边缘检测、人体姿态估计、深度估计等）来从输入参考图中提取出纯粹的控制信息，也就是我们常说的**条件图像**(Conditioning Image)。

当然的，我们也可以不使用预处理功能，直接输入一张自己处理好的图片当作预处理图。下面是Rocky构建的ControlNet的条件图像处理流程图示，让大家能够更好的理解：

![ControlNet的条件图像处理流程](./imgs/ControlNet的条件图像处理流程.png)

接着条件图像信息通过ControlNet再注入到Stable Diffusion和FLUX.1中，再加上原本就直接注入到Stable Diffusion和FLUX.1中的文本信息和图像信息（可选，进行图生图任务），综合作用进行扩散过程，最终生成受条件信息控制的图像。

总的来说，ControlNet做的就是这样一件事：**它为扩散模型（如 Stable Diffusion/FLUX.1）提供一种额外的“约束”条件，引导AIGC大模型按照我们期望的构图、姿态或结构来生成图像，减少图像生成的随机**性。

为了大家方便的理解，**Rocky也制作了ControlNet推理的完整流程图**，大家可以直观的学习理解：

![完整的ControlNet模型推理流程](./imgs/完整的ControlNet模型推理流程.png)

<h2 id="2.ControlNet的最小单元是什么样的？">2.ControlNet的最小单元是什么样的？</h2>

下图是ControlNet模型的最小单元：

![ControlNet模型的最小单元结构示意图](./imgs/ControlNet模型的最小单元结构示意图.png)

从上图可以看到，**在使用ControlNet模型之后，Stable Diffusion/FLUX.1模型的权重被复制出两个相同的部分，分别是“锁定”副本（locked）权重和“可训练”副本（trainable copy）权重**。

**我们如何理解这两个副本权重呢？** Rocky从训练角度和推理角度给大家进行通俗易懂的讲解。

**首先不管是训练阶段还是推理阶段，ControlNet都在“可训练”副本上输入控制条件** $c$，然后将“可训练”副本输出结果和原来Stable Diffusion/FLUX.1模型的“锁定”副本输出结果**相加（add）**获得最终的输出结果。

在训练阶段，**其中“锁定”副本中冻结参数，权重保持不变，保留了Stable Diffusion/FLUX.1模型原本的能力**；与此同时，**使用新数据对“可训练”副本进行微调训练，学习数据中的控制条件信息**。因为有Stable Diffusion/FLUX.1模型作为预训练权重，**复制“可训练”副本而不是直接训练原始权重还能避免数据集较小时的过拟合**，所以我们使用常规规模数据集（几K-几M级别）就能对控制条件进行学习训练，同时不会破坏Stable Diffusion/FLUX.1模型原本的能力（从数十亿张图像中学习到的大型模型的能力）。

另外，大家可能发现了**ControlNet模型的最小单元结构中有两个zero convolution模块，它们是1×1卷积，并且在微调训练时权重和偏置都初始化为零（zero初始化）**。这样一来，在我们开始训练ControlNet之前，所有zero convolution模块的输出都为零，使得ControlNet完完全全就在原有Stable Diffusion/FLUX.1底模型的能力上进行微调训练，这样可以尽量避免训练加入的初始噪声对ControlNet“可训练”副本权重的破坏，保证了不会产生大的能力偏差。

<h2 id="3.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？">3.ControlNet中的zero卷积层初始权重为什么是0?zero卷积层为什么有效？</h2>

大家很可能就会有一个疑问，如果zero convolution模块的初始权重为零，那么梯度也为零，ControlNet模型将不会学到任何东西。**那么为什么“zero convolution模块”有效呢？（AIGC算法面试必考点）**

Rocky进行下面的推导，相信大家对一切都会非常清晰明了：

我们可以假设ControlNet的初始权重为： $y=wx+b$ ，然后我们就可以得到对应的梯度求导：

$$\frac{\partial y}{\partial w}=x,\frac{\partial y}{\partial x}=w,\frac{\partial y}{\partial b}=1$$

如果此时 $w=0$ 并且 $x \neq 0$ ，然后我们就可以得到：

$$\frac{\partial y}{\partial w} \neq 0,\frac{\partial y}{\partial x}=0,\frac{\partial y}{\partial b}\neq 0$$

这就意味着只要 $x \neq 0$ ，一次梯度下降迭代将使w变成非零值。然后就得到： $\frac{\partial y}{\partial x}\neq 0$ 。**这样就能让zero convolution模块逐渐成为具有非零权重的卷积层，并不断优化参数权重**。


<h2 id="4.ControlNet系列模型有多少种控制条件？介绍一下各控制条件的原理与功能">4.ControlNet系列模型有多少种控制条件？介绍一下各控制条件的原理与功能</h2>

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

在实际使用中，我们经常会将多个ControlNet组合使用，以达到更复杂和精确的控制效果。


<h2 id="5.ConrtolNet是如何训练的？训练ControlNet模型的流程中有哪些关键参数？">5.ConrtolNet是如何训练的？训练ControlNet模型的流程中有哪些关键参数？</h2>

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

ControlNet系列模型的训练流程主要分成以下几个步骤：

1. 设计我们想要的额外控制条件：除了上面章节中讲到的控制条件，我们还可以根据实际需求自定义一些控制条件，从而使用ControlNet控制Stable Diffusion/FLUX.1朝着我们想要的细粒度方向生成内容。
2. 构建训练数据集：确定好额外控制条件后，我们就可以开始构建训练数据集了。ControlNet数据集中需要包含三个维度的信息：Ground Truth图片、作为控制条件（Conditional）的图片，以及对应的Caption标签。
3. 训练我们自己的ControlNet模型：训练数据集构建好后，我们就可以开始训练自己的ControlNet模型了，我们需要一个至少8G显存的GPU才能满足ControlNet模型的训练要求。


<h2 id="6.ControlNet的损失函数是什么？">6.ControlNet的损失函数是什么？</h2>


<h2 id="7.ControlNet中Balanced、My-prompt-is-more-important、ControlNet-is-more-important三种模式的区别是什么？">7.ControlNet中Balanced、My prompt is more important、ControlNet is more important三种模式的区别是什么？</h2>


<h2 id="8.ControlNet-1.1与ControlNet相比，有哪些改进？">8.ControlNet 1.1与ControlNet相比，有哪些改进？</h2>

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


<h2 id="9.介绍一下Controlnet-Union的原理">9.介绍一下Controlnet-Union的原理</h2>

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


<h2 id="10.ControlNet有哪些万金油级应用案例？">10.ControlNet有哪些万金油级应用案例？</h2>

---
