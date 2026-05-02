# 目录

## 第一章 数字人系统总览与技术路线

[1. AIGC 时代数字人是什么？核心技术栈如何划分？](#q-001)
  - [面试问题：2D 数字人、3D 数字人、实时交互数字人有什么区别？](#q-002)
  - [面试问题：一个完整数字人产品链路通常包括哪些模块？](#q-003)
  - [面试问题：数字人生成为什么同时需要身份、动作、音频、表情和时序一致性？](#q-004)

## 第二章 2D 数字人与可控人体生成

[2. 2D 数字人生成有哪些主流方向？](#q-005)
  - [面试问题：人体驱动、音频驱动、换脸、虚拟换衣、视频写真分别解决什么问题？](#q-006)
  - [面试问题：如何把图像生成模型扩展到视频数字人？](#q-007)
  - [面试问题：Animate Anyone 类人体驱动方法的核心结构是什么？](#q-008)
  - [面试问题：ReferenceNet、Pose Encoder、ControlNet、Motion Module 各自起什么作用？](#q-009)
  - [面试问题：人体驱动生成中如何提升脸部 ID 相似度？](#q-010)

## 第三章 音频驱动数字人与说话人视频生成

[3. 音频驱动数字人的核心难点是什么？](#q-011)
  - [面试问题：音频驱动说话人生成的一般 pipeline 是什么？](#q-012)
  - [面试问题：Wav2Vec、Whisper、HuBERT 等音频编码器在数字人里如何使用？](#q-013)
  - [面试问题：Hallo、Hallo2、Sonic、InfiniteTalk 的核心差异是什么？](#q-014)
  - [面试问题：为什么 Sonic 强调 Global Audio Perception？](#q-015)
  - [面试问题：长时长音频驱动视频如何避免漂移、闪烁和身份丢失？](#q-016)
  - [面试问题：多人音频驱动数字人为什么更难？](#q-017)

## 第四章 ID 保持、人脸定制与换脸

[4. ID 保持生成和换脸有什么本质区别？](#q-018)
  - [面试问题：IP-Adapter、InstantID、PuLID、PhotoMaker 的思路有什么区别？](#q-019)
  - [面试问题：如何实现高保真人脸身份注入？](#q-020)
  - [面试问题：换脸任务为什么常与 3DMM 人脸重建结合？](#q-021)
  - [面试问题：视频换脸如何保证时间一致性？](#q-022)
  - [面试问题：Copy-Paste 现象是什么？如何缓解？](#q-023)

## 第五章 虚拟换衣与角色替换

[5. 虚拟换衣在数字人中解决什么问题？](#q-024)
  - [面试问题：图像换衣和视频换衣有什么区别？](#q-025)
  - [面试问题：Mask-based 和 Mask-free 虚拟换衣路线有什么差异？](#q-026)
  - [面试问题：虚拟换衣如何构造训练三元组数据？](#q-027)
  - [面试问题：如何解决衣服纹理模糊、头发遮挡和体型差异问题？](#q-028)
  - [面试问题：Wan-Animate 类统一角色动画/替换框架的核心思想是什么？](#q-029)

## 第六章 3D 数字人、3D 表示与可微渲染

[6. 3D 数字人为什么仍然重要？](#q-030)
  - [面试问题：3DMM、FLAME、BFM 分别是什么？](#q-031)
  - [面试问题：SMPL、SMPL-X 的核心参数和区别是什么？](#q-032)
  - [面试问题：点云、Mesh、体素、SDF、NeRF、3D Gaussian Splatting 如何表示 3D 人体/人脸？](#q-033)
  - [面试问题：传统渲染管线和可微渲染有什么区别？](#q-034)
  - [面试问题：单目 RGB 重建 3D 人体/人脸时有哪些难点？](#q-035)

## 第七章 训练数据、评估指标与工程落地

[7. 数字人项目如何从数据到部署落地？](#q-036)
  - [面试问题：数字人训练数据如何采集、清洗和标注？](#q-037)
  - [面试问题：音频驱动人脸数据如何预处理？](#q-038)
  - [面试问题：数字人质量如何评估？](#q-039)
  - [面试问题：实时交互数字人的低延迟链路如何设计？](#q-040)
  - [面试问题：Duix-Mobile 这类移动端实时数字人的工程价值是什么？](#q-041)
  - [面试问题：数字人安全、合规和反滥用需要注意什么？](#q-042)

---

<h1 id="q-001">1. AIGC 时代数字人是什么？核心技术栈如何划分？</h1>

数字人是将人物外观、身份、语音、表情、动作、交互能力和场景表现整合到一起的虚拟人系统。AIGC 时代的数字人不再只是传统动画或 3D 建模，而是由大模型、扩散/视频生成模型、语音模型、3D 表示、实时渲染和交互式 Agent 共同驱动。

从技术栈看，数字人可以拆成六层：

1. **感知层**：人脸检测、关键点、头姿、人体姿态、分割、深度、音频特征、情绪识别。
2. **身份层**：人脸 ID embedding、外观 reference、服装/发型/配饰、角色一致性建模。
3. **驱动层**：音频驱动、文本驱动、姿态驱动、视频驱动、表情驱动、多角色驱动。
4. **生成层**：GAN、Diffusion、Video Diffusion、DiT、Flow Matching、NeRF/3DGS、可微渲染。
5. **交互层**：ASR、LLM、RAG、TTS、情绪控制、打断、流式对话。
6. **工程层**：低延迟推理、移动端部署、视频编码、流媒体、缓存、量化、端云协同。

<h2 id="q-002">面试问题：2D 数字人、3D 数字人、实时交互数字人有什么区别？</h2>

**2D 数字人**通常以图像或视频为输出，核心是让一个参考人物在目标音频、目标姿态或目标文本条件下动起来。它更依赖扩散视频生成、talking head、pose-guided animation、换脸和换衣技术。

**3D 数字人**通常显式建模几何、骨骼、材质、表情和渲染，可以在任意视角和场景中驱动。它更依赖 3DMM、FLAME、SMPL/SMPL-X、Blendshape、LBS、NeRF/3DGS 和渲染引擎。

**实时交互数字人**强调对话闭环和低延迟体验，通常包括：

$$
\text{用户语音} \rightarrow \text{ASR} \rightarrow \text{LLM} \rightarrow \text{TTS} \rightarrow \text{唇形/表情驱动} \rightarrow \text{渲染播放}
$$

**区别总结：**

- 2D 数字人：视觉生成效果强，适合视频内容生产。
- 3D 数字人：可控性强，适合游戏、直播、XR、虚拟拍摄。
- 实时交互数字人：系统工程要求高，核心指标是延迟、稳定性、可打断和对话自然度。

<h2 id="q-003">面试问题：一个完整数字人产品链路通常包括哪些模块？</h2>

一个完整数字人链路可以分为离线生产和在线交互两类。

**离线视频生产链路：**

1. 输入角色素材：参考图、人像视频、服装图、音频、脚本、目标动作。
2. 预处理：人脸检测、裁剪、对齐、分割、姿态估计、音频降噪、说话人分离。
3. 条件编码：ID embedding、pose map、DensePose、landmark、audio token、text embedding。
4. 视频生成：U-Net/DiT/Video Diffusion/Flow 模型生成连续帧。
5. 后处理：超分、脸部修复、插帧、色彩一致性、背景融合、音视频合成。
6. 质检：唇形同步、身份一致性、闪烁、手部质量、内容安全。

**实时交互链路：**

1. ASR 流式识别用户语音。
2. LLM/Agent 生成响应。
3. TTS 流式合成音频。
4. 数字人 SDK 根据音频流驱动唇形、表情和动作。
5. 前端渲染、播放和打断控制。

<h2 id="q-004">面试问题：数字人生成为什么同时需要身份、动作、音频、表情和时序一致性？</h2>

数字人不像普通视频生成，只要求“画面合理”。用户会对人脸、嘴型、眼神、动作和身份非常敏感，因此需要同时满足多个一致性：

- **身份一致性**：脸像不像同一个人。
- **唇音一致性**：嘴型是否和音素、节奏匹配。
- **表情一致性**：情绪是否和语音内容、语调一致。
- **动作一致性**：头部、身体、手势是否自然。
- **时序一致性**：视频是否闪烁、漂移、身份逐渐变化。
- **场景一致性**：光照、背景、服装、遮挡关系是否稳定。

这些条件之间还会冲突。例如强行提升 ID 相似度可能降低表情动态，强音频 CFG 可能让嘴型更准但脸部变形，强姿态控制可能导致服装纹理拉伸。因此数字人模型设计的核心是多条件控制之间的平衡。

---

<h1 id="q-005">2. 2D 数字人生成有哪些主流方向？</h1>

2D 数字人生成主要围绕“用少量素材生成可控人物图像/视频”。在 AIGC 时代，主流方向包括可控人体动画、音频驱动 talking portrait、ID 保持人像生成、换脸、虚拟换衣、视频写真、多人角色动画等。

<h2 id="q-006">面试问题：人体驱动、音频驱动、换脸、虚拟换衣、视频写真分别解决什么问题？</h2>

**人体驱动**：输入参考人物图像和目标姿态序列，生成参考人物按目标动作运动的视频。典型条件包括 DWPose、OpenPose、DensePose、Depth、Normal、SMPL/SMPL-X。

**音频驱动**：输入参考头像/视频和音频，生成与音频同步的说话人视频。核心难点是唇形、表情、头动和身份保持。

**换脸**：把源身份迁移到目标视频中的脸部，同时保留目标视频姿态、表情、光照和背景。

**虚拟换衣**：将指定服装穿到目标人物身上，保留人体、姿态、场景和服装细节。

**视频写真/ID 保持视频生成**：输入一个或多个参考 ID 图像，生成同一人物在不同场景、动作、镜头下的视频。

**面试总结：**

人体驱动强调姿态控制，音频驱动强调音画同步，换脸强调局部身份替换，虚拟换衣强调服装几何和纹理迁移，视频写真强调人物身份和画面美感的长期一致。

<h2 id="q-007">面试问题：如何把图像生成模型扩展到视频数字人？</h2>

从图像生成扩展到视频生成，关键是让模型具备时序建模能力。常见路线有三类：

**1. 逐帧生成 + 后处理**

每帧独立生成，再用光流、插帧、时序平滑或人脸修复减轻闪烁。优点是实现简单，缺点是时序一致性差。

**2. 图像扩散模型 + Motion Module**

在 Stable Diffusion/SDXL 等图像模型基础上插入时间模块，例如 AnimateDiff 风格的 temporal attention / motion module，让模型处理多帧 latent：

$$
z \in \mathbb{R}^{B \times C \times T \times H \times W}
$$

这种方式能复用图像模型的空间生成能力，同时学习视频运动先验。

**3. 原生视频生成模型/DiT/Flow**

使用 Stable Video Diffusion、Wan、HunyuanVideo、DiT/Rectified Flow 等视频底模，把人像、音频、姿态等条件注入视频 latent。优点是视频一致性更强，缺点是训练和推理成本更高。

<h2 id="q-008">面试问题：Animate Anyone 类人体驱动方法的核心结构是什么？</h2>

Animate Anyone 类方法通常由三部分组成：

1. **ReferenceNet**：提取参考图像外观细节，包括脸、衣服、发型、纹理、背景。
2. **Pose Encoder**：编码目标姿态序列，如 DWPose/OpenPose/DensePose。
3. **Denoising U-Net + Motion Module**：在扩散去噪过程中融合参考外观和目标姿态，并通过时序模块保证帧间一致。

抽象训练目标是：

$$
L=\mathbb{E}_{z_t,t,\epsilon,c}\left[\|\epsilon-\epsilon_\theta(z_t,t,c_{\text{ref}},c_{\text{pose}})\|_2^2\right]
$$

其中 $c_{\text{ref}}$ 是参考图条件，$c_{\text{pose}}$ 是姿态条件。

**面试要点：**

- ReferenceNet 负责“像谁、穿什么”。
- Pose Encoder 负责“怎么动”。
- Motion Module 负责“连续帧是否稳定”。
- CLIP image embedding 可提供全局语义，但细粒度纹理通常需要 ReferenceNet 或局部特征。

<h2 id="q-009">面试问题：ReferenceNet、Pose Encoder、ControlNet、Motion Module 各自起什么作用？</h2>

**ReferenceNet**：通常是与 U-Net 相似的结构，用参考图提取多尺度特征，在去噪网络中通过 attention 或 feature injection 注入外观信息。

**Pose Encoder**：把姿态图、骨架图、DensePose、Depth、Normal 等结构条件编码成 latent feature，用来约束人物动作。

**ControlNet**：在冻结原扩散模型的基础上，增加可训练控制分支，使模型遵循边缘、深度、姿态等空间条件。优点是兼容性好，缺点是视频时序控制需要额外设计。

**Motion Module**：建模时间维度上的相邻帧关系，常见形式包括 temporal convolution、temporal attention、3D U-Net block。

**条件注入方式：**

- concat 到 latent channel
- cross attention
- self attention feature injection
- residual adapter
- control branch
- token 拼接或 DiT 条件调制

<h2 id="q-010">面试问题：人体驱动生成中如何提升脸部 ID 相似度？</h2>

脸部 ID 相似度难提升，原因是脸部区域小、人类敏感度高、动作/表情变化大，而且扩散模型容易优先满足姿态和整体画面。

常见方案：

1. **人脸 ID Loss**：使用 ArcFace/InsightFace 等人脸识别模型提取生成脸和参考脸 embedding，约束余弦相似度。

$$
L_{\text{id}}=1-\cos(f_{\text{face}}(I_{\text{gen}}),f_{\text{face}}(I_{\text{ref}}))
$$

2. **Face Adapter**：单独提取人脸区域特征，并在 U-Net/DiT 中注入。

3. **面部 mask 加权**：对脸部区域增加重建损失、感知损失或 attention 权重。

4. **局部高分辨率修复**：生成后对人脸区域做 face restoration 或局部重绘。

5. **多参考图训练/推理**：用多角度、多表情图像提取更稳健 ID 表示，减少 copy-paste。

---

<h1 id="q-011">3. 音频驱动数字人的核心难点是什么？</h1>

音频驱动数字人的目标是根据语音生成自然说话视频。它不仅要让嘴型同步，还要让头部、眼神、表情、身体和情绪与音频一致，同时保持身份和背景稳定。

<h2 id="q-012">面试问题：音频驱动说话人生成的一般 pipeline 是什么？</h2>

典型 pipeline：

1. 输入参考图或参考视频。
2. 音频预处理：重采样、降噪、人声分离、切片、音量归一化。
3. 音频编码：提取 wav2vec/Whisper/HuBERT/mel 特征。
4. 视觉预处理：人脸检测、裁剪对齐、landmark、face mask、head pose。
5. 条件融合：将音频 token 与视觉 latent 通过 cross attention、adapter 或 concat 融合。
6. 视频生成：扩散/视频扩散/DiT 逐段生成。
7. 后处理：超分、脸部增强、音视频 mux、时序平滑。

核心映射可以写成：

$$
V=\mathcal{G}(I_{\text{ref}}, A, c_{\text{pose}}, c_{\text{id}})
$$

其中 $A$ 是音频条件，$I_{\text{ref}}$ 是参考身份，$c_{\text{pose}}$ 是姿态/头动条件，$c_{\text{id}}$ 是身份条件。

<h2 id="q-013">面试问题：Wav2Vec、Whisper、HuBERT 等音频编码器在数字人里如何使用？</h2>

音频编码器负责把波形转成与语音内容、节奏、音素、情绪相关的特征。

**Wav2Vec 2.0**：常用于提取连续语音表征，适合音频驱动唇形、表情和头动。

**Whisper**：强在语音识别和多语种鲁棒性，部分方法用其 encoder feature 表达语音语义和节奏。

**HuBERT**：通过隐藏单元预测学习语音离散结构，适合表达音素级信息。

**Mel Spectrogram**：传统而高效，适合实时系统和轻量模型。

音频特征通常需要和视频帧率对齐。若音频特征帧率为 $F_a$，视频帧率为 $F_v$，需要插值、池化或窗口聚合：

$$
a_t = \text{Pool}(A_{[t-\Delta,t+\Delta]})
$$

**面试要点：**

- 唇形更依赖局部音素。
- 表情和头动更依赖长窗口语义、节奏和情绪。
- 长视频更需要全局音频感知，否则容易每段局部正确但整体动作割裂。

<h2 id="q-014">面试问题：Hallo、Hallo2、Sonic、InfiniteTalk 的核心差异是什么？</h2>

**Hallo**：基于 Stable Diffusion/AnimateDiff 思路做音频驱动肖像动画。核心包括 wav2vec 音频特征、face locator、image/audio projection、motion module 等，强调分层音频驱动视觉合成。

**Hallo2**：面向长时长、高分辨率音频驱动肖像动画。相较 Hallo，重点增强长视频稳定性和高分辨率输出，并引入视频超分/脸部增强等工程链路。

**Sonic**：强调 Global Audio Perception，即不能只看短窗口音频，还要感知全局语音节奏、情绪和结构。其本地项目使用 Stable Video Diffusion img2vid-xt、Whisper tiny、audio2token/audio2bucket、RIFE 等组件。

**InfiniteTalk**：面向 sparse-frame video dubbing 和无限长度视频生成，支持 video-to-video 与 image-to-video。它不仅同步嘴唇，也尝试让头动、身体姿态和表情与音频一致。本地 README 中显示其基于 Wan2.1-I2V-14B、中文 wav2vec2、流式/clip 模式、TeaCache、int8 量化、多 GPU 推理等工程设计。

**一句话对比：**

Hallo 偏经典扩散 talking portrait，Hallo2 强化长时长和高分辨率，Sonic 强调全局音频感知，InfiniteTalk 强调稀疏帧配音、视频到视频、无限长度和工程推理能力。

<h2 id="q-015">面试问题：为什么 Sonic 强调 Global Audio Perception？</h2>

传统 audio-driven talking head 往往用局部音频窗口驱动当前帧：

$$
x_t = G(I_{\text{ref}}, A_{t-k:t+k})
$$

这对唇形同步足够，但对表情、头动、情绪和长句节奏不够。人说话时，点头、停顿、强调、情绪变化往往依赖更长时间范围的语音结构。

Global Audio Perception 的直觉是让模型看到更长的音频上下文：

$$
x_t = G(I_{\text{ref}}, A_{1:T}, t)
$$

这样模型不仅知道当前音素，还知道当前处在一句话的开头、强调处、停顿处还是收尾处。

**收益：**

- 头动和表情更自然。
- 音频情绪与面部表达更一致。
- 长视频中动作节奏不容易碎片化。

<h2 id="q-016">面试问题：长时长音频驱动视频如何避免漂移、闪烁和身份丢失？</h2>

长视频问题通常来自逐段生成误差累积。常见方案：

1. **分块生成 + 重叠融合**：每段生成时与上一段保留重叠帧，通过光流或 latent blending 平滑过渡。
2. **参考帧/稀疏帧约束**：周期性注入原视频帧或关键帧，防止身份和背景漂移。
3. **全局音频上下文**：避免每个片段只根据局部音频独立运动。
4. **时间位置编码偏移**：类似长视频 temporal offset，让模型知道当前片段的全局时间位置。
5. **ID/Face embedding 持续注入**：每段都注入稳定身份条件。
6. **视频超分和修复后处理**：如 Hallo2 中使用 RealESRGAN/CodeFormer 类组件提升高分辨率质量。
7. **低显存/加速策略**：TeaCache、量化、FSDP、多 GPU，保证长视频能实际跑完。

<h2 id="q-017">面试问题：多人音频驱动数字人为什么更难？</h2>

多人场景比单人难在“音频-人物绑定”和“空间-身份一致”。

核心挑战：

- 多路音频可能重叠，需要说话人分离或 diarization。
- 每个人的嘴型应该只跟自己的音频同步。
- 画面中人物位置会变化，需要稳定追踪 ID。
- 多人交互有遮挡、转头、远近变化和动作交叉。
- 文本提示、音频和人物 mask 之间要绑定正确。

常见方案：

1. 为每个人维护独立 ID embedding、face mask、audio feature。
2. 使用人物区域 mask 或 tracking ID 做条件路由。
3. 使用 Label RoPE / instance embedding 区分不同人物 token。
4. 多人数据训练中显式构造“谁在说话”的监督信号。
5. 在生成后用人脸检测和唇形同步指标逐人质检。

---

<h1 id="q-018">4. ID 保持生成和换脸有什么本质区别？</h1>

ID 保持生成是在生成过程中让输出人物保持参考身份，但可以改变场景、风格、服装、姿态和表情；换脸则通常是在目标图像/视频中替换脸部身份，同时尽量保留目标的姿态、表情、光照、背景和非脸部区域。

<h2 id="q-019">面试问题：IP-Adapter、InstantID、PuLID、PhotoMaker 的思路有什么区别？</h2>

**IP-Adapter**：在扩散模型中增加图像提示能力，使用 CLIP image encoder 提取 image prompt，通过解耦 cross attention 注入图像条件，冻结原模型，训练轻量 adapter。

**InstantID**：使用人脸识别模型提取强 ID embedding，并结合面部关键点/IdentityNet 控制身份和结构，适合单图 ID 保持生成。

**PuLID**：强调 Pure and Lightning ID customization，希望在保持 ID 的同时尽量不破坏原 T2I 模型能力。其思路包括轻量 T2I 分支、ID loss、对比对齐损失等。

**PhotoMaker**：通过多张同 ID 图像构造 stacked ID embedding，让模型学习更稳定的身份表示，减少单图参考里的姿态、表情、光照干扰。

**对比：**

- IP-Adapter 更通用，图像提示不只限人脸。
- InstantID 更偏强 ID 注入和结构控制。
- PuLID 更强调不影响原模型可编辑性。
- PhotoMaker 更强调多图身份聚合。

<h2 id="q-020">面试问题：如何实现高保真人脸身份注入？</h2>

高保真身份注入通常需要三类信息：

1. **身份 embedding**：ArcFace/InsightFace 提取身份向量。
2. **结构条件**：landmark、face parsing、depth、3DMM、head pose。
3. **细节参考**：局部纹理、肤色、发型、配饰等图像特征。

注入方式：

- cross attention 中注入 ID token。
- 在 key/value 中加入 ID adapter。
- 使用面部 mask 让 ID 条件只影响脸部区域。
- 在多尺度 U-Net/DiT 层中注入 face feature。
- 使用 ID loss、perceptual loss、face region loss 训练。

**关键平衡：**

ID 注入太弱会不像本人；太强会 copy-paste，导致姿态、表情和风格可编辑性下降。

<h2 id="q-021">面试问题：换脸任务为什么常与 3DMM 人脸重建结合？</h2>

换脸需要在目标视频中保留表情、头姿和光照，同时替换身份。2D 图像特征容易受姿态和表情影响，3DMM 可以把人脸拆成更可控的参数：

$$
S = \bar{S} + A_{\text{id}}\alpha + A_{\text{exp}}\beta
$$

其中 $\alpha$ 表示身份形状，$\beta$ 表示表情。

结合 3DMM 的好处：

- 显式估计头姿、表情、形状。
- 更好地处理大角度侧脸和遮挡。
- 能提供人脸 mask、法线、深度、UV 纹理等几何条件。
- 有助于目标表情迁移到源身份。

<h2 id="q-022">面试问题：视频换脸如何保证时间一致性？</h2>

视频换脸的一致性包括 ID 稳定、边界稳定、光照稳定和表情连续。

常用方法：

1. **时序模型**：使用 3D 卷积、temporal attention、ConvLSTM 或视频扩散模块。
2. **光流约束**：相邻帧生成结果经光流 warp 后应接近当前帧。
3. **身份一致损失**：所有帧的人脸 embedding 应接近同一源身份。
4. **边界融合稳定**：mask 边界平滑，避免脸部区域跳变。
5. **关键帧传播**：高质量关键帧结果传播到相邻帧。
6. **后处理**：颜色匹配、泊松融合、视频去闪烁。

<h2 id="q-023">面试问题：Copy-Paste 现象是什么？如何缓解？</h2>

Copy-Paste 指模型不是学习“身份在不同姿态/表情/光照下如何保持”，而是直接复制参考脸的局部外观，导致输出看起来像贴图，表情僵硬、角度不自然、可编辑性差。

原因：

- 训练数据多为同图重建，缺少同 ID 不同姿态/表情配对。
- ID 条件过强，模型把非身份因素也编码进 ID token。
- 缺少显式解耦，表情、姿态、光照与身份混在一起。

缓解方法：

- 多图同 ID 数据训练。
- DropToken/DropPath 解耦 ID 与结构细节。
- 对比损失约束有无 ID 条件路径。
- 延迟 ID 注入，前期生成结构，后期补身份细节。
- 使用 face parsing/mask 限制 ID 条件作用区域。

---

<h1 id="q-024">5. 虚拟换衣在数字人中解决什么问题？</h1>

虚拟换衣让数字人或真实人物穿上指定服装，是电商试穿、虚拟模特、短视频生产和角色替换的重要能力。它的核心是保持人体身份、姿态和场景，同时迁移服装形状、纹理、材质和遮挡关系。

<h2 id="q-025">面试问题：图像换衣和视频换衣有什么区别？</h2>

图像换衣只需要单帧合理，而视频换衣还要保证多帧一致。

视频换衣额外挑战：

- 服装纹理不能闪烁。
- 衣服边界要随人体动作稳定变化。
- 遮挡关系要一致，如头发、手臂、包、外套。
- 布料动态要自然，如裙摆、袖子、褶皱。
- 长视频中颜色和细节不能漂移。

**面试总结：**

图像换衣是空间一致性问题，视频换衣是空间一致性 + 时间一致性 + 非刚性布料运动问题。

<h2 id="q-026">面试问题：Mask-based 和 Mask-free 虚拟换衣路线有什么差异？</h2>

**Mask-based** 方法把换衣区域视作 inpainting 区域：

$$
I_{\text{out}} = G(I_{\text{person}}\odot(1-M), I_{\text{cloth}}, M, c)
$$

优点是目标区域明确；缺点是依赖人体解析和 mask 质量，mask 过大容易丢失人体/头发/手臂信息，mask 过小则衣服替换不完整。

**Mask-free** 方法不显式依赖二值换衣 mask，而是通过服装编码、人体姿态和注意力对齐来生成穿着效果。优点是保留原图信息更好，缺点是对齐更难，训练数据要求更高。

**实践中常用混合方案：**

粗 mask 提供区域先验，服装 encoder 提供纹理，pose/densepose 提供几何，扩散模型负责细节生成和融合。

<h2 id="q-027">面试问题：虚拟换衣如何构造训练三元组数据？</h2>

典型训练三元组：

$$
(I_{\text{person}}, I_{\text{cloth}}, I_{\text{target}})
$$

其中：

- $I_{\text{person}}$：目标人物图像，通常需要去除或遮挡原衣服区域。
- $I_{\text{cloth}}$：商品服装图或平铺图。
- $I_{\text{target}}$：人物真实穿着该服装的图像。

构造方式：

1. 同一模特同一服装的商品图和上身图配对。
2. 从电商数据中清洗服装 SKU、模特图、平铺图。
3. 使用人体解析提取衣服区域和 agnostic person representation。
4. 过滤遮挡严重、姿态极端、服装不可见、图文不匹配样本。
5. 视频换衣还需要同一服装跨帧稳定标注或通过跟踪自动构造。

<h2 id="q-028">面试问题：如何解决衣服纹理模糊、头发遮挡和体型差异问题？</h2>

**衣服纹理模糊：**

- 使用高分辨率训练或多尺度训练。
- 引入服装局部特征 encoder。
- 对纹理区域增加感知损失或 patch loss。
- 推理时分块超分或局部重绘。

**头发遮挡衣服：**

- 使用人体/头发/手臂 parsing mask。
- 建模层级：背景 < 衣服 < 手/头发/配饰。
- 对遮挡边界做 alpha blending 或局部 inpainting。

**体型差异：**

- 使用 DensePose/SMPL 对齐人体表面。
- 使用 pose retargeting 调整骨骼比例。
- 使用文本或控制信号描述宽松/紧身/长短。
- 对服装形状和人体形状分别编码，避免模型把原衣服轮廓泄露到结果中。

<h2 id="q-029">面试问题：Wan-Animate 类统一角色动画/替换框架的核心思想是什么？</h2>

统一角色动画/替换框架希望同一个模型同时支持：

- 角色动画：让源角色按照驱动动作运动。
- 角色替换：把目标视频中的人物替换成源角色，同时保留环境和动作。

核心思想是把外观、姿态、面部、环境拆成不同条件：

$$
V_{\text{out}} = G(I_{\text{src}}, V_{\text{drive}}, C_{\text{body}}, C_{\text{face}}, C_{\text{env}})
$$

常见模块：

- **Body Adapter**：编码驱动视频的人体姿态、动作、DensePose/骨架。
- **Face Adapter**：编码面部区域表情、眼神和细粒度变化。
- **Relighting LoRA**：让替换角色适应目标环境光照和色调。
- **Pose Retargeting**：处理源角色与驱动角色体型差异。

**面试要点：**

这类方法的难点不只是“让人动”，而是保持源角色外观、目标动作、环境光照、遮挡关系和长视频稳定性同时成立。

---

<h1 id="q-030">6. 3D 数字人为什么仍然重要？</h1>

虽然 2D 生成模型效果越来越强，3D 数字人仍然重要，因为它提供可控视角、物理一致性、骨骼驱动、实时渲染和交互空间能力。游戏、XR、直播、虚拟拍摄、工业仿真和长时交互仍高度依赖 3D 表示。

<h2 id="q-031">面试问题：3DMM、FLAME、BFM 分别是什么？</h2>

**3DMM** 是三维可变形模型，通过统计人脸数据建立形状和纹理基：

$$
S = \bar{S} + A_s\alpha,\quad T=\bar{T}+A_t\beta
$$

通过调整低维参数生成不同人脸。

**BFM** 是经典 3DMM 人脸模型，包含人脸形状和纹理空间，适合人脸重建和纹理拟合。

**FLAME** 是更现代的人头模型，覆盖头部、脖子、眼球和表情，常用于表情动画、头部重建和 neural rendering。相比 BFM，FLAME 更强调可动画性和头部结构；BFM 更强调人脸纹理统计空间。

**应用：**

- 单图/视频 3D 人脸重建
- 表情迁移
- 换脸几何对齐
- 头部姿态估计
- 可微渲染监督

<h2 id="q-032">面试问题：SMPL、SMPL-X 的核心参数和区别是什么？</h2>

SMPL 是参数化 3D 人体模型，输入形状参数 $\beta$ 和姿态参数 $\theta$，输出人体 mesh。

常见设定：

- 形状参数 $\beta$：控制高矮胖瘦等体型。
- 姿态参数 $\theta$：控制 24 个关节旋转。
- 输出约 6890 个顶点和 24 个关节。

核心技术：

- shape blend shapes
- pose blend shapes
- Linear Blend Skinning（LBS）

SMPL-X 是 SMPL 的扩展，统一建模身体、手和脸：

- 顶点更多。
- 关节包含手指、脸部、眼球等。
- 增加面部表情和手部姿态参数。

**区别：**

SMPL 适合全身粗粒度人体动作；SMPL-X 更适合数字人，因为手势、表情和眼神对真实感非常关键。

<h2 id="q-033">面试问题：点云、Mesh、体素、SDF、NeRF、3D Gaussian Splatting 如何表示 3D 人体/人脸？</h2>

**点云**：一组 3D 点：

$$
P=\{(x_i,y_i,z_i)\}_{i=1}^{N}
$$

采集容易，但缺少拓扑连接。

**Mesh**：顶点、边、面组成的表面，适合渲染、动画和物理绑定，是传统数字人常用表示。

**体素**：三维规则网格，类似 3D 像素。简单但分辨率高时内存大。

**SDF**：有符号距离函数：

$$
f(x,y,z)=d
$$

表面为：

$$
f(x,y,z)=0
$$

适合隐式曲面和可微优化。

**NeRF**：学习从 3D 坐标和视角到密度、颜色的函数：

$$
F_\theta(x,d)=(\sigma,c)
$$

优点是新视角合成质量高，缺点是传统 NeRF 渲染慢、动画控制难。

**3D Gaussian Splatting**：用大量带颜色、透明度、协方差的 3D 高斯表示场景，渲染速度快，适合实时新视角合成。数字人中可用于头部 avatar、动态场景、实时渲染。

<h2 id="q-034">面试问题：传统渲染管线和可微渲染有什么区别？</h2>

传统渲染从 3D 场景到 2D 图像：

$$
I = R(M,A,P,L)
$$

其中 $M$ 是几何，$A$ 是材质/外观，$P$ 是相机，$L$ 是光照。

传统管线包括：

- 顶点变换
- 投影
- 光栅化
- Z-buffer
- 着色

问题在于光栅化和 Z-buffer 涉及离散选择，例如某像素是否在三角形内部、哪个三角形离相机最近，这些操作天然不可导。

可微渲染希望得到：

$$
\frac{\partial I}{\partial M},\quad \frac{\partial I}{\partial A},\quad \frac{\partial I}{\partial P},\quad \frac{\partial I}{\partial L}
$$

常见方式：

- 对不可导步骤做软化近似。
- 设计可微 rasterizer。
- 使用神经渲染替代部分传统管线。

**数字人中的作用：**

可微渲染让 2D 图像监督反向优化 3D 人脸/人体参数、材质、光照和表情。

<h2 id="q-035">面试问题：单目 RGB 重建 3D 人体/人脸时有哪些难点？</h2>

单目 RGB 缺少真实深度，因此是病态问题。

主要难点：

- 深度歧义：同一 2D 投影可能对应多个 3D 姿态。
- 遮挡：手、头发、衣服、物体会遮挡关键部位。
- 光照和纹理混淆：阴影可能被误认为几何。
- 表情和身份耦合：脸型、表情、姿态相互影响。
- 宽松衣物：SMPL 类裸体/紧身模型难表示裙摆、袖子、布料褶皱。

改进方向：

- 多帧时序约束。
- 多模态输入，如 RGB-D、IMU、多视角。
- 隐式场或神经表面表示。
- SMPL/FLAME 先验 + diffusion/生成模型补细节。
- 物理约束和布料模拟。

---

<h1 id="q-036">7. 数字人项目如何从数据到部署落地？</h1>

数字人项目落地不仅取决于模型效果，还取决于数据、延迟、稳定性、成本、可控性和安全。面试中常问的不只是“某模型原理”，还包括“如何做一套能上线的数字人系统”。

<h2 id="q-037">面试问题：数字人训练数据如何采集、清洗和标注？</h2>

高质量数字人数据通常要满足：

- 人脸清晰，分辨率足够。
- 音频清楚，噪声低，音画同步。
- 姿态、表情、动作多样。
- 身份一致，避免混入相似人物。
- 光照、背景、镜头变化可控。
- 授权和合规明确。

常见预处理：

1. 视频抽帧和质量筛选。
2. 人脸检测、裁剪、对齐。
3. landmark、head pose、face parsing、body pose。
4. 音频 VAD、人声分离、重采样、响度归一化。
5. 音画同步检测，过滤延迟样本。
6. 身份聚类，剔除错 ID。
7. NSFW、版权、未授权人脸过滤。

<h2 id="q-038">面试问题：音频驱动人脸数据如何预处理？</h2>

音频驱动数据预处理重点是“音画同步”和“脸部稳定”。

步骤：

1. 用 VAD 切分有声片段。
2. 转为统一采样率，如 16kHz。
3. 人声分离或降噪，减少背景音乐干扰。
4. 检测人脸 bbox 和关键点。
5. 根据人脸中心裁剪，保持脸部占画面 50%-70%。
6. 过滤侧脸过大、遮挡严重、多人脸混入样本。
7. 提取 wav2vec/Whisper 特征，并与视频帧时间戳对齐。
8. 计算 SyncNet/LSE 指标，过滤音画不同步样本。

Hallo/Hallo2 本地 README 也强调输入图应为正脸、方形裁剪、脸部为主体，音频为清晰 WAV。

<h2 id="q-039">面试问题：数字人质量如何评估？</h2>

数字人评估要分维度，不应只看 FID。

**身份一致性：**

$$
\cos(f_{\text{id}}(I_{\text{gen}}),f_{\text{id}}(I_{\text{ref}}))
$$

**唇形同步：**

- SyncNet confidence
- LSE-D / LSE-C
- 音素-口型对齐准确率

**视频质量：**

- FID/FVD
- LPIPS
- 人脸清晰度
- 超分后细节质量

**时序一致性：**

- optical flow consistency
- flicker score
- identity variance across frames

**动作质量：**

- 关键点误差
- 手部关键点置信度
- 姿态跟随误差

**主观评估：**

- ID 像不像
- 嘴型准不准
- 表情自然不自然
- 是否有恐怖谷
- 是否可用于商业场景

<h2 id="q-040">面试问题：实时交互数字人的低延迟链路如何设计？</h2>

实时交互数字人的核心指标是端到端延迟。链路通常是：

$$
\text{Mic} \rightarrow \text{ASR} \rightarrow \text{LLM} \rightarrow \text{TTS} \rightarrow \text{Avatar} \rightarrow \text{Render}
$$

优化方向：

- ASR 流式识别，边说边出 partial result。
- LLM 流式输出，首 token 延迟低。
- TTS 流式合成，边生成边播放。
- 数字人端支持流式音频驱动，不等整句音频。
- 预加载 avatar 资源和模型权重。
- 本地唇形/表情推理，减少网络往返。
- 支持打断 barge-in，用户说话时停止当前播报。
- 使用轻量模型、量化、缓存、端云协同。

<h2 id="q-041">面试问题：Duix-Mobile 这类移动端实时数字人的工程价值是什么？</h2>

Duix-Mobile 本地资源定位是可部署在手机或嵌入式屏幕上的实时对话数字人 SDK，支持 Android/iOS，可集成 LLM、ASR、TTS，强调本地低延迟和流式音频。

工程价值：

- 适合客服、政务、教育、医疗、车载、IoT、大屏等交互场景。
- 本地处理降低弱网影响。
- 可把数字人作为 UI 层，接入任意 LLM/ASR/TTS 服务。
- 响应延迟和流式播放比离线视频生成更关键。
- 移动端部署需要模型轻量化、资源预加载、音频流驱动和稳定渲染。

**与生成式视频数字人的区别：**

Duix-Mobile 这类 SDK 更偏实时交互和产品集成；Hallo/Sonic/InfiniteTalk 更偏高质量离线或半离线音频驱动视频生成。

<h2 id="q-042">面试问题：数字人安全、合规和反滥用需要注意什么？</h2>

数字人技术涉及人脸、声音和身份，合规风险很高。

需要关注：

- 肖像权、声音权、版权授权。
- 明确告知 AI 生成或 AI 驱动。
- 防止未经授权换脸、诈骗、冒充公众人物。
- 对生成内容加水印或溯源标识。
- 对敏感行业输出做审计和留痕。
- 数据采集要有授权，训练集要可追溯。
- 面向用户的定制数字人要有删除和撤回机制。

**面试回答重点：**

数字人不是单纯技术问题。越接近真人、越可实时交互，越需要安全策略、身份授权、内容审核和平台治理共同设计。

---

## 参考资料

[1] Xu et al. *Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation*. 2024. <https://arxiv.org/abs/2406.08801>

[2] Cui et al. *Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation*. ICLR 2025. <https://arxiv.org/abs/2410.07718>

[3] Ji et al. *Sonic: Shifting Focus to Global Audio Perception in Portrait Animation*. CVPR 2025.

[4] Yang et al. *InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing*. 2025. <https://arxiv.org/abs/2508.14033>

[5] Hu. *Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation*. 2023.

[6] Guo et al. *LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control*. 2024.

[7] Blanz and Vetter. *A Morphable Model for the Synthesis of 3D Faces*. SIGGRAPH 1999.

[8] Li et al. *FLAME: Learning a Model of Facial Shape and Expression from 4D Scans*. SIGGRAPH Asia 2017.

[9] Loper et al. *SMPL: A Skinned Multi-Person Linear Model*. SIGGRAPH Asia 2015.

[10] Pavlakos et al. *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image*. CVPR 2019.

[11] Mildenhall et al. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis*. ECCV 2020.

[12] Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. SIGGRAPH 2023.
