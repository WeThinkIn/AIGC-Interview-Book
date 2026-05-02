# 目录

## 第一章 GAN 数字人总览

[1. GAN 在数字人发展中解决过什么核心问题？](#q-001)
  - [面试问题：GAN、Diffusion、DiT/Flow 在数字人生成中的差异是什么？](#q-002)
  - [面试问题：为什么现在仍然需要学习 GAN 数字人？](#q-003)
  - [面试问题：GAN 数字人的典型模块有哪些？](#q-004)

## 第二章 基于 GAN 的 2D 数字人

[2. 基于 GAN 的 2D 数字人有哪些主流方向？](#q-005)
  - [面试问题：StyleGAN 系列为什么是人脸生成的重要基础？](#q-006)
  - [面试问题：StyleGAN 的 latent space 为什么适合做人脸编辑和数字人定制？](#q-007)
  - [面试问题：GAN Inversion 在数字人里有什么作用？](#q-008)
  - [面试问题：FOMM、Monkey-Net、TPSMM 这类一阶运动模型如何驱动肖像？](#q-009)
  - [面试问题：LivePortrait 为什么仍采用 GAN/隐式关键点路线也能获得强实时效果？](#q-010)
  - [面试问题：GAN Talking Head 如何做音频驱动口型？](#q-011)

## 第三章 GAN 换脸、ID 保持与视频一致性

[3. 基于 GAN 的换脸为什么长期占据实时应用？](#q-012)
  - [面试问题：DeepFakes、FaceSwap、FaceShifter、SimSwap、HiFiFace 的核心差异是什么？](#q-013)
  - [面试问题：ArcFace 身份特征如何注入 GAN 换脸网络？](#q-014)
  - [面试问题：GAN 换脸如何处理姿态、表情、遮挡和边界融合？](#q-015)
  - [面试问题：GAN 视频数字人如何保证时序一致性？](#q-016)
  - [面试问题：GAN 换脸相比 Diffusion 换脸还有哪些优势和短板？](#q-017)

## 第四章 基于 GAN 的 3D 数字人与 3D-aware 生成

[4. 什么是 3D-aware GAN？](#q-018)
  - [面试问题：HoloGAN、pi-GAN、GIRAFFE、StyleNeRF、EG3D 如何演进？](#q-019)
  - [面试问题：EG3D 的 tri-plane 表示为什么重要？](#q-020)
  - [面试问题：3D-aware GAN 如何实现人脸视角一致性？](#q-021)
  - [面试问题：3D-aware GAN 与 NeRF/3DGS Avatar 有什么区别？](#q-022)
  - [面试问题：GAN 如何用于 3D 数字人的纹理、材质和细节增强？](#q-023)

## 第五章 AIGC 时代数字人前沿趋势

[5. AIGC 数字人从 GAN 走向了哪些新范式？](#q-024)
  - [面试问题：扩散式人体动画和 GAN 人体动画的本质差别是什么？](#q-025)
  - [面试问题：OmniHuman、Wan-Animate、StableAnimator、SkyReels-A1 代表什么趋势？](#q-026)
  - [面试问题：Hallo/Hallo2/Hallo3、Sonic、InfiniteTalk、EMO、VASA、MuseTalk 如何分类？](#q-027)
  - [面试问题：DiT、Flow Matching、多模态大视频模型对数字人有什么影响？](#q-028)
  - [面试问题：AIGC 数字人为什么越来越强调多条件混合训练？](#q-029)
  - [面试问题：实时交互数字人与离线高保真数字人的技术路线如何取舍？](#q-030)

## 第六章 高频综合题与工程落地

[6. 面试中如何系统回答“设计一个数字人生成系统”？](#q-031)
  - [面试问题：如何评估 GAN/Diffusion 数字人的质量？](#q-032)
  - [面试问题：GAN 数字人线上常见问题如何排查？](#q-033)
  - [面试问题：数字人安全、版权和反滥用为什么越来越重要？](#q-034)
  - [面试问题：当前 AI 数字人论文和 GitHub 项目可以如何分层跟踪？](#q-035)
  - [面试问题：HeyGen、Synthesia、Tavus、D-ID、Akool 等产品体现了哪些落地方向？](#q-036)
  - [面试问题：如何建立 AI 数字人方向的持续学习清单？](#q-037)

---

<h1 id="q-001">1. GAN 在数字人发展中解决过什么核心问题？</h1>

GAN，全称 Generative Adversarial Network，是数字人生成早期最重要的生成模型之一。它通过生成器和判别器的对抗训练，让生成器学习产生接近真实图像或视频的结果。

GAN 在数字人中主要解决过四类问题：

1. **高保真人脸生成**

   StyleGAN 系列显著提升了人脸图像的真实感、分辨率和可编辑性，使“虚拟人脸生成、身份编辑、年龄/表情/姿态编辑”成为可用能力。

2. **人脸换脸与身份迁移**

   DeepFakes、FaceShifter、SimSwap、HiFiFace 等方法通过 identity embedding、attribute encoder、融合网络实现高效换脸。

3. **肖像动画与表情驱动**

   FOMM、face reenactment、GAN talking head 等方法用关键点、光流、3DMM 或音频特征驱动静态头像动起来。

4. **3D-aware 人脸生成**

   HoloGAN、pi-GAN、StyleNeRF、EG3D 等方法将 GAN 与 3D 表示、体渲染或 tri-plane 结合，提升视角一致性。

从技术演进看，GAN 数字人是连接传统图像编辑、神经渲染和当前扩散/DiT 数字人的重要阶段。

<h2 id="q-002">面试问题：GAN、Diffusion、DiT/Flow 在数字人生成中的差异是什么？</h2>

| 维度 | GAN | Diffusion | DiT/Flow Matching |
| --- | --- | --- | --- |
| 生成方式 | 一步或少步前向生成 | 多步去噪生成 | Transformer/Flow 建模连续生成路径 |
| 优势 | 推理快、图像锐利、实时友好 | 质量高、多样性强、条件控制灵活 | 扩展性强，适合大规模视频/多模态底模 |
| 短板 | 训练不稳定，mode collapse，复杂条件难 | 推理慢，时序一致性需额外设计 | 训练资源大，工程门槛高 |
| 数字人应用 | 换脸、实时肖像动画、StyleGAN 编辑 | 人体动画、口播视频、换衣、ID 保持 | 大视频模型、全身数字人、长视频、多条件生成 |
| 典型模型 | StyleGAN、SimSwap、FOMM、EG3D | Animate Anyone、Hallo、EMO、StableAnimator | OmniHuman、Wan-Animate、Hallo3、Sora/Veo 类范式 |

面试金句：

GAN 擅长低延迟和清晰局部生成，Diffusion 擅长高保真和复杂条件，DiT/Flow 代表数字人向大规模视频基础模型迁移。

<h2 id="q-003">面试问题：为什么现在仍然需要学习 GAN 数字人？</h2>

虽然 AIGC 数字人主流研究大量转向扩散模型和 DiT，但 GAN 仍值得学习，原因有四点：

1. **很多线上系统仍在使用 GAN**

   实时换脸、低延迟人脸驱动、移动端肖像动画、直播美颜和局部增强常常需要毫秒级推理，GAN 仍有工程优势。

2. **GAN 是理解身份注入和属性解耦的基础**

   ID embedding、attribute encoder、latent editing、GAN inversion、identity loss 等思想被后续扩散模型继承。

3. **3D-aware GAN 是 3D 生成的重要先导**

   EG3D 的 tri-plane、StyleNeRF 的风格化神经辐射场、pi-GAN 的隐式场生成，都影响了后续 3D 生成和 Avatar 表示。

4. **GAN 与 Diffusion 可以组合**

   GAN 可用于超分、人脸修复、细节增强、判别器辅助训练、蒸馏和实时生成。

所以面试中不能简单说“GAN 过时了”。更准确的说法是：GAN 在通用高保真长视频生成上让位于 Diffusion/DiT，但在实时、局部、轻量、可编辑场景仍然有价值。

<h2 id="q-004">面试问题：GAN 数字人的典型模块有哪些？</h2>

GAN 数字人系统通常包含以下模块：

- **生成器 Generator**：根据 latent code、身份特征、姿态/表情/音频条件生成图像。
- **判别器 Discriminator**：判断图像是否真实，也可判断身份、局部区域、时序片段是否合理。
- **身份编码器 ID Encoder**：常用 ArcFace/InsightFace 提取人脸身份特征。
- **属性编码器 Attribute Encoder**：提取目标图像的姿态、表情、光照、背景等非身份属性。
- **运动估计模块 Motion Estimator**：估计关键点、光流、dense motion 或 deformation field。
- **融合模块 Fusion/Blending**：把生成脸与目标背景、头发、遮挡区域自然融合。
- **损失函数**：GAN loss、identity loss、perceptual loss、reconstruction loss、feature matching loss、landmark loss、temporal loss。

典型训练目标可以写成：

$$
\mathcal{L} =
\lambda_{\text{adv}}\mathcal{L}_{\text{adv}} +
\lambda_{\text{id}}\mathcal{L}_{\text{id}} +
\lambda_{\text{rec}}\mathcal{L}_{\text{rec}} +
\lambda_{\text{perc}}\mathcal{L}_{\text{perc}} +
\lambda_{\text{temp}}\mathcal{L}_{\text{temp}}
$$

---

<h1 id="q-005">2. 基于 GAN 的 2D 数字人有哪些主流方向？</h1>

基于 GAN 的 2D 数字人主要包括：

1. **人脸生成**

   使用 StyleGAN/StyleGAN2/StyleGAN3 生成高清虚拟人脸，用于虚拟角色、数据增强和身份编辑。

2. **人脸编辑**

   在 StyleGAN latent space 中编辑年龄、性别、表情、姿态、发型、眼镜、妆容等属性。

3. **GAN Inversion**

   将真实人脸图像反演到 GAN 潜空间，从而对真实人物进行编辑或重建。

4. **肖像动画**

   输入一张参考头像和一段驱动视频/音频，生成对应表情、头动和口型。

5. **换脸**

   把源身份迁移到目标图像/视频，同时保留目标姿态、表情、光照和背景。

6. **人脸修复与超分**

   使用 GFPGAN、CodeFormer、Real-ESRGAN 类方法提升数字人视频脸部和纹理质量。

<h2 id="q-006">面试问题：StyleGAN 系列为什么是人脸生成的重要基础？</h2>

StyleGAN 系列之所以重要，是因为它把 GAN 人脸生成从“能生成”推进到“高质量、可控、可编辑”。

核心思想包括：

- **Mapping Network**：把随机噪声 $z$ 映射到更解耦的中间潜空间 $w$。
- **Style Modulation**：用 style 控制不同层级的生成特征。
- **逐层控制**：低层控制姿态和脸型，中层控制五官，高层控制纹理和颜色。
- **噪声注入**：控制头发、皮肤纹理等随机细节。
- **Path length regularization**：改善潜空间平滑性。
- **Alias-free 设计**：StyleGAN3 进一步减少纹理粘连和视角变化伪影。

在数字人中，StyleGAN 的价值是：

- 生成高质量虚拟身份。
- 提供可编辑潜空间。
- 支持真实人脸反演和属性编辑。
- 可与 3DMM、语义分割、CLIP、文本控制结合。

<h2 id="q-007">面试问题：StyleGAN 的 latent space 为什么适合做人脸编辑和数字人定制？</h2>

StyleGAN 的潜空间具有相对平滑和语义解耦的特点。常见潜空间包括：

- $Z$：原始随机噪声空间。
- $W$：经过 mapping network 后的中间空间，通常更解耦。
- $W+$：每一层使用不同 $w$，表达能力更强，适合图像反演。
- $S$：style space，直接对应调制参数，部分属性更局部可控。

人脸编辑通常是在 latent space 中寻找语义方向：

$$
w' = w + \alpha d
$$

其中 $d$ 可以表示年龄、微笑、性别、姿态、眼镜、发型等方向。

数字人定制中的用途：

- 用 GAN inversion 得到真实人物 latent。
- 用语义方向编辑人物属性。
- 用身份损失保持人物不变。
- 用插值生成身份过渡或表情变化。

风险：

- 编辑方向可能不完全解耦，例如“变老”同时改变发型和肤色。
- $W+$ 重建强但可编辑性可能下降。
- 对 out-of-domain 图像反演效果不稳定。

<h2 id="q-008">面试问题：GAN Inversion 在数字人里有什么作用？</h2>

GAN Inversion 是把一张真实图像映射回 GAN 潜空间的过程：

$$
w^* = \arg\min_w \mathcal{L}(G(w), I)
$$

常见方法：

1. **优化式反演**

   直接优化 latent code，重建质量较高但速度慢。

2. **编码器式反演**

   训练 encoder 一次前向得到 latent，速度快但细节可能差。

3. **混合式反演**

   先用 encoder 初始化，再进行少量优化。

4. **PTI/Pivotal Tuning**

   在给定人物上微调生成器，提升重建保真度。

数字人用途：

- 真实人脸编辑。
- 虚拟人身份创建。
- 人脸修复和超分。
- 文本/属性控制的人脸重绘。
- 将真实人物转换到可控生成空间。

面试要点：GAN Inversion 的核心矛盾是“重建保真度”和“可编辑性”。重建越强，可能越依赖 $W+$ 或生成器微调；但越偏离原始 GAN 分布，后续编辑越容易产生伪影。

<h2 id="q-009">面试问题：FOMM、Monkey-Net、TPSMM 这类一阶运动模型如何驱动肖像？</h2>

这类方法的核心是：从源图像和驱动视频中估计稀疏或密集运动，再把源图像的外观按照驱动运动变形到目标姿态。

典型流程：

1. **关键点估计**

   从源图和驱动帧中估计无监督关键点或局部仿射变换。

2. **运动场估计**

   根据关键点差异生成 dense motion field。

3. **特征变形**

   将源图特征 warp 到驱动姿态。

4. **遮挡预测**

   预测哪些区域不可见，需要网络补全。

5. **图像生成**

   通过生成器输出最终头像动画帧。

公式上可以理解为：

$$
I_t = G(I_s, \mathcal{W}_{s \rightarrow t}, O_t)
$$

其中 $I_s$ 是源图，$\mathcal{W}_{s \rightarrow t}$ 是运动场，$O_t$ 是遮挡图。

优点：

- 不需要明确的人脸 3D 标注。
- 泛化到不同物体类别的潜力较强。
- 推理速度相对快。

缺点：

- 大姿态和大遮挡下容易崩。
- 对身份细节和口腔内部建模不足。
- 长视频稳定性依赖运动估计质量。

<h2 id="q-010">面试问题：LivePortrait 为什么仍采用 GAN/隐式关键点路线也能获得强实时效果？</h2>

LivePortrait 代表了一类非常工程化的肖像动画路线：不追求大型扩散模型的全局生成能力，而是把问题聚焦到“给定参考肖像，快速、稳定、可控地迁移表情和头动”。

它的关键思想可以概括为：

- **隐式关键点表示**：不只依赖传统 2D landmark，而是学习更适合动画驱动的关键点/运动表示。
- **运动变换和特征 warping**：将源图特征根据驱动运动变形。
- **拼接与重定向控制**：对眼睛、嘴巴、表情幅度、头动等做可控调整。
- **轻量生成器**：相比扩散模型多步采样，GAN/encoder-decoder 式生成更适合实时。
- **工程后处理**：人脸裁剪、贴回、边界融合、口型/眨眼 retargeting 都影响最终体验。

它说明一个重要事实：数字人不是所有场景都要用最大的模型。对于实时头像动画，强约束任务加轻量模型可能比通用视频生成模型更实用。

<h2 id="q-011">面试问题：GAN Talking Head 如何做音频驱动口型？</h2>

GAN Talking Head 通常把音频特征转为嘴部运动、表情或中间结构，再生成视频帧。

常见路线：

1. **Audio-to-Landmark**

   音频编码器预测嘴部 landmark，再用图像生成器渲染人脸。

2. **Audio-to-3DMM**

   音频预测 3DMM/FLAME 表情和下颌参数，再渲染条件图驱动生成器。

3. **Audio-to-Feature**

   音频特征直接注入生成网络中，与身份图像特征融合。

4. **局部嘴部修复**

   只重绘嘴部区域，保留原视频的头部、背景和身份。

常见损失：

- 重建损失：保证帧内容接近真实。
- 对抗损失：提升嘴部和脸部真实感。
- 同步损失：使用 SyncNet 类模型约束音画同步。
- 身份损失：保持人物身份。
- 感知损失：提升视觉质量。

早期 GAN talking head 速度快，但容易出现口腔细节假、情绪弱、头动单一、长视频累积误差等问题。当前扩散/DiT 方法则更强调全脸表情、头部动作、身体手势和长时序一致性。

---

<h1 id="q-012">3. 基于 GAN 的换脸为什么长期占据实时应用？</h1>

GAN 换脸长期占据实时应用，核心原因是推理速度快、结构清晰、工程链路成熟。

典型换脸目标是：

$$
I_{\text{out}} = G(I_{\text{target}}, e_{\text{id}}^{\text{source}})
$$

其中：

- $I_{\text{target}}$ 提供目标姿态、表情、光照、背景和头发。
- $e_{\text{id}}^{\text{source}}$ 提供源人物身份。
- $G$ 输出保留目标属性但具有源身份的人脸。

GAN 换脸通常只重绘脸部局部区域，因此比整帧视频扩散生成更轻量。它在直播、短视频、移动端、批量视频处理里有明显速度优势。

<h2 id="q-013">面试问题：DeepFakes、FaceSwap、FaceShifter、SimSwap、HiFiFace 的核心差异是什么？</h2>

| 方法 | 核心思路 | 优势 | 局限 |
| --- | --- | --- | --- |
| DeepFakes/AutoEncoder | 为源/目标身份训练自编码器或共享编码器 | 概念简单，早期效果直观 | 每个身份需训练，泛化弱 |
| FaceSwap 传统路线 | 检测对齐、人脸重建、颜色融合 | 工程可控 | 高保真和大姿态有限 |
| FaceShifter | 身份编码 + 属性编码 + 自适应融合 | 一次训练可泛化多身份 | 遮挡和边界需额外处理 |
| SimSwap | ArcFace 身份特征 + 弱特征匹配 | 高效、泛化好、实时友好 | 极端角度和细节仍有限 |
| HiFiFace | 强调 3D 形状感知和高保真身份 | 细节和身份更强 | 模型与训练更复杂 |

面试总结：

早期 DeepFakes 是 per-identity 训练；FaceShifter/SimSwap 进入 one-shot 泛化换脸；HiFiFace 等方法进一步增强 3D 几何和高保真细节。

<h2 id="q-014">面试问题：ArcFace 身份特征如何注入 GAN 换脸网络？</h2>

ArcFace 这类人脸识别模型可以提取源人脸身份 embedding：

$$
e_{\text{id}} = E_{\text{ArcFace}}(I_{\text{source}})
$$

注入方式常见有：

1. **AdaIN/Style Modulation**

   将身份 embedding 映射为通道级缩放和平移，调制生成器特征。

2. **Feature Concatenation**

   将身份向量扩展到空间维度，与目标特征拼接。

3. **Attention Injection**

   让目标特征通过注意力读取身份 token。

4. **Identity Adaptive Normalization**

   类似条件归一化，用身份控制生成过程。

训练时常用 identity loss：

$$
\mathcal{L}_{\text{id}} = 1 - \cos(E(I_{\text{out}}), E(I_{\text{source}}))
$$

但身份注入不能过强，否则会破坏目标表情、姿态和光照，出现 copy-paste 或融合不自然。好的换脸系统需要平衡：

- 源身份相似度。
- 目标属性保持。
- 局部融合自然度。
- 视频时序稳定性。

<h2 id="q-015">面试问题：GAN 换脸如何处理姿态、表情、遮挡和边界融合？</h2>

常见策略：

1. **姿态与表情保持**

   目标图像通过 attribute encoder 提供姿态、表情、光照和背景特征，源图只提供身份。

2. **3DMM 辅助对齐**

   使用 3DMM 估计头姿和脸部几何，做姿态归一化、mask 生成、UV 对齐或形状约束。

3. **遮挡处理**

   对头发、眼镜、手、麦克风等遮挡区域预测 occlusion mask，避免生成脸覆盖不可替换区域。

4. **边界融合**

   使用 segmentation mask、Poisson blending、颜色匹配、alpha matte 或专门的 blending network。

5. **局部判别器**

   对眼睛、嘴巴、脸部边界等区域使用局部 adversarial loss，提升细节。

6. **多尺度生成**

   先生成低分辨率结构，再补高分辨率细节，减少边界伪影。

面试中要强调：换脸不是简单替换五官，而是身份与属性解耦。源身份应该进入脸型和五官身份特征，目标属性应该保留姿态、表情、光照和上下文。

<h2 id="q-016">面试问题：GAN 视频数字人如何保证时序一致性？</h2>

GAN 视频数字人常见时序一致性策略：

1. **输入条件稳定**

   对 landmark、3DMM、pose、mask 做时序平滑，减少逐帧抖动。

2. **光流约束**

   用 optical flow 将前一帧 warp 到当前帧，约束生成结果一致。

3. **时序判别器**

   判别器不只看单帧，也看连续帧 clip，判断运动是否真实。

4. **循环一致性**

   正向驱动和反向重建保持一致，减少身份漂移。

5. **特征缓存**

   在在线推理中缓存身份特征、背景特征和上一帧状态。

6. **后处理平滑**

   对颜色、边界、关键点和生成区域做 temporal filter。

GAN 的优势是单帧生成速度快，短时间内帧间差异小；但如果输入驱动不稳，逐帧生成仍然会闪烁。因此工程上“预处理稳定性”往往和模型本身同样重要。

<h2 id="q-017">面试问题：GAN 换脸相比 Diffusion 换脸还有哪些优势和短板？</h2>

| 维度 | GAN 换脸 | Diffusion 换脸/ID 保持 |
| --- | --- | --- |
| 推理速度 | 快，适合实时 | 慢，需蒸馏或加速 |
| 局部保真 | 脸部清晰，局部任务强 | 真实感高但可能改动过多 |
| 可控性 | 结构固定，工程可控 | 条件丰富但可能不稳定 |
| 泛化能力 | 对极端姿态和遮挡有限 | 大模型泛化更强 |
| 编辑能力 | 主要局部身份替换 | 可同时改风格、场景、服装 |
| 时序一致性 | 短视频较稳，依赖驱动 | 长视频需时序模型/参考记忆 |
| 主要风险 | 边界伪影、身份不够像 | copy-paste、身份漂移、推理成本高 |

结论：

- 实时直播换脸、移动端、批量视频处理：GAN 仍有优势。
- 高质量创意生成、复杂 prompt、全身角色重绘：Diffusion/DiT 更强。
- 产品级系统常用混合方案：GAN 做局部实时，Diffusion 做高质量离线或增强。

---

<h1 id="q-018">4. 什么是 3D-aware GAN？</h1>

3D-aware GAN 是指在生成图像时显式或隐式引入三维结构，使生成结果在视角变化下保持几何一致。

普通 2D GAN 只学习图像分布：

$$
z \rightarrow I
$$

3D-aware GAN 则试图学习：

$$
z, c_{\text{camera}} \rightarrow I
$$

同时内部有某种 3D 表示，例如 voxel、feature volume、implicit field、NeRF、tri-plane 或 mesh-aware feature。

它的目标不是只生成一张好看人脸，而是生成一个可以从不同相机视角渲染、身份和几何相对一致的人脸或人体。

<h2 id="q-019">面试问题：HoloGAN、pi-GAN、GIRAFFE、StyleNeRF、EG3D 如何演进？</h2>

| 方法 | 关键思想 | 贡献 |
| --- | --- | --- |
| HoloGAN | 学习 3D feature volume，再投影成 2D 图像 | 早期无监督 3D-aware GAN |
| pi-GAN | 用 SIREN/隐式神经场表示 radiance field | 将 GAN 与隐式 3D 表示结合 |
| GIRAFFE | 组合式神经特征场，分离物体和背景 | 支持场景组合与可控相机 |
| StyleNeRF | StyleGAN 风格控制 + NeRF 表示 | 兼顾高分辨率风格生成和 3D 一致性 |
| EG3D | tri-plane 表示 + neural rendering + super-resolution | 大幅提升 3D-aware 人脸生成质量和效率 |

演进主线：

1. 从 2D 图像 GAN 走向 3D-aware feature。
2. 从显式体素走向隐式场和体渲染。
3. 从低分辨率 neural rendering 走向高分辨率超分。
4. 从视角可控走向几何更稳定、身份更一致。

<h2 id="q-020">面试问题：EG3D 的 tri-plane 表示为什么重要？</h2>

EG3D 的核心是 tri-plane representation。它不用完整 3D 体素网格，而是使用三个正交的二维特征平面：

- $F_{xy}$
- $F_{xz}$
- $F_{yz}$

对于三维点 $x=(x,y,z)$，分别投影到三个平面上采样特征，再聚合得到该点的特征：

$$
f(x) = F_{xy}(x,y) + F_{xz}(x,z) + F_{yz}(y,z)
$$

再通过轻量 decoder 得到颜色和密度，用体渲染生成低分辨率图像，最后通过超分网络得到高分辨率结果。

tri-plane 的优势：

- 比 3D voxel 更省内存。
- 比纯 MLP NeRF 查询更快。
- 与 StyleGAN2 生成器兼容，能继承高质量图像先验。
- 能保持一定 3D 一致性。

它的重要性在于：为高质量 3D-aware 人脸生成提供了效率和质量之间的平衡点。

<h2 id="q-021">面试问题：3D-aware GAN 如何实现人脸视角一致性？</h2>

3D-aware GAN 通常通过以下机制实现视角一致：

1. **显式相机条件**

   生成时输入相机位姿，让模型学习同一 latent identity 在不同视角下的投影。

2. **三维中间表示**

   使用 feature volume、NeRF、tri-plane 等表示，让图像不是直接生成，而是从 3D 表示渲染出来。

3. **体渲染或投影**

   对相机射线采样三维点，计算颜色和密度，保证多视角共享同一空间表示。

4. **身份 latent 共享**

   同一个 latent code 对应同一人物，不同相机只改变观察角度。

5. **pose-aware 训练**

   训练中引入相机分布、姿态估计或镜像增强，避免模型把姿态误当成身份。

局限：

- 多视角一致性通常仍弱于真实 3D 扫描或 NeRF/3DGS Avatar。
- 背面头部、耳朵、头发等区域可能出现幻觉。
- 对超大姿态、复杂发型和遮挡仍不稳定。

<h2 id="q-022">面试问题：3D-aware GAN 与 NeRF/3DGS Avatar 有什么区别？</h2>

| 维度 | 3D-aware GAN | NeRF/3DGS Avatar |
| --- | --- | --- |
| 目标 | 学习人脸/人体类别分布，能随机生成新身份 | 针对一个或多个具体人物重建可渲染 Avatar |
| 输入 | 随机 latent、相机、条件 | 多视角图像/视频、相机标定、驱动参数 |
| 输出 | 新身份图像或可视角控制图像 | 具体人物的新视角/动态渲染 |
| 训练方式 | 对抗训练，类别级生成 | 重建/渲染损失，实例级或少量身份 |
| 编辑性 | latent editing 较强 | 几何/外观编辑依赖表示 |
| 一致性 | 有 3D 先验但可能不完美 | 对捕获人物通常更一致 |

简洁回答：

3D-aware GAN 更像“会生成很多 3D 感虚拟人的生成器”；NeRF/3DGS Avatar 更像“为某个真实人物建立一个可渲染数字分身”。

<h2 id="q-023">面试问题：GAN 如何用于 3D 数字人的纹理、材质和细节增强？</h2>

GAN 不一定只生成整张脸，也可以作为 3D 数字人的细节增强模块。

常见用法：

1. **UV 纹理生成**

   在 UV 空间生成或补全人脸/人体纹理，保证贴图可绑定到 Mesh。

2. **法线/位移细节**

   生成皮肤皱纹、毛孔、衣物褶皱对应的 normal map 或 displacement map。

3. **人脸修复**

   对渲染后的人脸区域使用 GFPGAN/CodeFormer 类模型增强五官和皮肤细节。

4. **超分辨率**

   对低分辨率渲染帧做超分，降低实时渲染压力。

5. **域适配**

   将 3D 渲染图从 CG 风格转为真实视频风格。

6. **判别器辅助训练**

   在神经渲染或扩散训练中加入 adversarial loss，提升清晰度和局部真实感。

风险：

- GAN 可能 hallucinate 身份细节，导致不像本人。
- 局部增强可能破坏时序一致性。
- UV 空间和图像空间增强需要注意接缝和投影关系。

---

<h1 id="q-024">5. AIGC 数字人从 GAN 走向了哪些新范式？</h1>

AIGC 数字人正在从“局部生成和身份编辑”走向“多条件、长视频、全身、可交互、多模态”的新范式。

主要变化：

1. **从 GAN 到 Diffusion**

   从一步生成转为多步去噪，提升多样性、文本可控性和高保真复杂场景生成能力。

2. **从 U-Net Diffusion 到 DiT/Flow**

   大模型架构从卷积 U-Net 走向 Transformer 和 Flow Matching，更适合大规模视频和多模态条件。

3. **从头像到全身**

   任务从 talking head 扩展到半身、全身、手势、物体交互、舞蹈和多人场景。

4. **从单条件到多条件混合**

   同时利用文本、音频、姿态、参考图、视频、深度、3D 参数等条件。

5. **从短视频到长视频**

   强调跨分钟级甚至小时级的一致性、身份保持和记忆机制。

6. **从离线生成到实时交互**

   ASR、LLM、TTS、数字人驱动和渲染形成低延迟闭环。

<h2 id="q-025">面试问题：扩散式人体动画和 GAN 人体动画的本质差别是什么？</h2>

| 维度 | GAN 人体动画 | 扩散式人体动画 |
| --- | --- | --- |
| 生成机制 | 直接生成或 warping + refinement | 从噪声逐步去噪生成视频 |
| 条件控制 | 姿态、关键点、光流、身份图像 | 文本、姿态、参考图、音频、深度、分割等 |
| 真实感 | 局部锐利，依赖训练域 | 综合真实感更高 |
| 多样性 | 较低，容易模式受限 | 更强 |
| 时序建模 | 光流/时序判别器/缓存 | motion module、temporal attention、3D VAE、video DiT |
| 推理速度 | 快 | 慢，需要加速 |
| 工程适配 | 实时强 | 离线高质量强 |

扩散模型的优势是能利用大规模图像/视频预训练和丰富条件控制；GAN 的优势是推理快、局部任务稳定。当前高质量人体动画通常偏扩散/DiT，实时应用仍可能保留 GAN 或轻量生成器。

<h2 id="q-026">面试问题：OmniHuman、Wan-Animate、StableAnimator、SkyReels-A1 代表什么趋势？</h2>

这些方法代表 AIGC 数字人向“统一角色动画和多条件视频生成”发展。

**OmniHuman**：

- 强调多条件混合训练。
- 支持图像、文本、音频、姿态等条件组合。
- 从头部说话扩展到全身动作和手势。
- 核心趋势是利用更大规模、更复杂条件的数据训练统一模型。

**Wan-Animate**：

- 强调角色动画和角色替换统一。
- 既可以让参考角色跟随目标动作，也可以替换视频中的角色。
- 代表从单一 pose transfer 走向角色级视频编辑。

**StableAnimator**：

- 强调身份保持、姿态控制和稳定动画生成。
- 通常结合参考图、姿态序列和视频扩散结构。

**SkyReels-A1**：

- 偏向高质量人像/人物视频生成与动画控制。
- 体现大视频模型在人类动作生成中的落地趋势。

共同趋势：

- 从单一头像到全身人物。
- 从单条件到多条件。
- 从短片段到长视频。
- 从研究 demo 到内容生产工具。
- 从 U-Net 扩散向 DiT/视频基础模型迁移。

<h2 id="q-027">面试问题：Hallo/Hallo2/Hallo3、Sonic、InfiniteTalk、EMO、VASA、MuseTalk 如何分类？</h2>

可以按任务目标和模型路线分类：

| 方法 | 主要任务 | 路线特点 |
| --- | --- | --- |
| MuseTalk | 实时/高效唇形同步 | 局部嘴部生成，工程速度友好 |
| EMO | 音频到表情丰富肖像视频 | Audio2Video diffusion，强调情绪和表现力 |
| VASA | 实时音频驱动 talking face | 强调低延迟、头动和表情动态 |
| Hallo | 音频驱动肖像动画 | Stable Diffusion/AnimateDiff 风格，分层音频视觉合成 |
| Hallo2 | 长时长高分辨率肖像动画 | 强化长视频稳定和高分辨率 |
| Hallo3 | DiT 架构肖像动画探索 | 从 U-Net/AnimateDiff 走向 Diffusion Transformer |
| Sonic | 音频驱动肖像/人体动态 | 强调 Global Audio Perception |
| InfiniteTalk | 长时长视频配音和音频驱动 | 稀疏帧、视频到视频、无限长度推理 |

面试回答逻辑：

- **只修嘴**：MuseTalk 类，速度快但全局动态有限。
- **头像动画**：Hallo/EMO/VASA 类，关注头动、表情、口型。
- **长视频**：Hallo2/InfiniteTalk 类，关注分段一致性和跨段记忆。
- **DiT/大模型**：Hallo3/OmniHuman 类，代表架构升级。
- **全局音频感知**：Sonic 类，强调音频不仅控制嘴，还影响头部、表情和身体。

<h2 id="q-028">面试问题：DiT、Flow Matching、多模态大视频模型对数字人有什么影响？</h2>

DiT、Flow Matching 和大视频模型正在改变数字人的模型底座。

影响包括：

1. **更强的长程时序建模**

   Transformer 更适合建模长视频 token 之间的关系，缓解短窗口扩散模型的时序断裂。

2. **更强的条件融合**

   文本、图像、音频、姿态、深度、3D 参数可以统一为 token 或 latent 条件。

3. **更好的可扩展性**

   DiT/Flow 模型可以随着数据和算力扩展，接近通用视频生成底模路线。

4. **更自然的全身运动**

   大视频模型学到更多人体运动、物理交互和镜头语言，减少只会“动嘴”的问题。

5. **更高的工程成本**

   训练和推理资源更大，需要蒸馏、缓存、量化、并行和分辨率分级。

对数字人来说，未来路线很可能是：

$$
\text{大视频底模} + \text{身份/姿态/音频/3D 条件适配器} + \text{实时/离线分级部署}
$$

<h2 id="q-029">面试问题：AIGC 数字人为什么越来越强调多条件混合训练？</h2>

数字人不是单条件任务。真实视频中的动作同时受文本语义、音频节奏、情绪、身份、姿态、镜头、场景和交互对象影响。

多条件混合训练的价值：

1. **扩大可用数据**

   严格的音频-姿态-身份全标注数据很少，但单独的图像、视频、音频、姿态数据很多。混合训练可以利用更多弱条件数据。

2. **提升泛化能力**

   模型同时见过文本生成、图像动画、音频驱动、姿态控制，更容易适应不同输入组合。

3. **减少条件过拟合**

   如果只训练音频到视频，模型可能只会动嘴；加入姿态/文本/视频条件能学习更完整的人体动态。

4. **支持产品灵活性**

   同一个模型可以支持只给音频、给音频+姿态、给参考视频、给文本脚本等多种使用方式。

5. **增强鲁棒性**

   某个条件缺失或质量差时，模型仍能借助其他条件生成合理结果。

训练难点：

- 不同条件强度不一致，容易互相冲突。
- 数据质量和标注粒度不同。
- 需要条件 dropout、任务采样比例、条件路由和损失平衡。

<h2 id="q-030">面试问题：实时交互数字人与离线高保真数字人的技术路线如何取舍？</h2>

| 维度 | 实时交互数字人 | 离线高保真数字人 |
| --- | --- | --- |
| 核心指标 | 低延迟、稳定、可打断 | 真实感、表现力、分辨率 |
| 常用模型 | 3D Avatar、BlendShape、轻量 GAN、局部唇形模型 | Diffusion、DiT、NeRF/3DGS、视频增强 |
| 输入 | ASR/LLM/TTS 流式输出 | 脚本、音频、参考图、姿态、镜头 |
| 输出 | 实时渲染帧 | 高质量视频 |
| 延迟要求 | 百毫秒到秒级 | 可接受分钟级或更长 |
| 风险 | 口型延迟、表情僵硬、交互中断 | 生成慢、身份漂移、长视频闪烁 |

取舍原则：

- 客服、直播互动、移动端：优先实时链路，牺牲部分真实感。
- 广告、影视、短视频内容生产：优先高保真离线生成。
- 高端产品：实时 3D 驱动负责交互，离线 AIGC 负责高质量内容资产。

---

<h1 id="q-031">6. 面试中如何系统回答“设计一个数字人生成系统”？</h1>

可以按以下框架回答：

1. **明确场景**

   是实时客服、口播视频、全身动画、换脸、虚拟换衣，还是 3D Avatar？

2. **确定输入条件**

   参考图、音频、文本、姿态、视频、服装、相机、背景、3D 参数分别有哪些？

3. **选择技术路线**

   - 实时头像：LivePortrait/MuseTalk/GAN/BlendShape。
   - 高质量口播：Hallo/EMO/Sonic/InfiniteTalk。
   - 全身人物：Animate Anyone/OmniHuman/Wan-Animate。
   - 3D 交互：SMPL-X/FLAME/Mesh/3DGS。
   - 换脸：GAN 换脸或 Diffusion ID 保持。

4. **设计模型结构**

   身份编码器、条件编码器、运动模块、生成器/去噪器、渲染器、后处理模块。

5. **设计训练目标**

   重建、对抗、感知、身份、同步、姿态、时序、遮挡、正则和安全约束。

6. **设计工程链路**

   预处理、推理、缓存、超分、编码、审核、监控和回退策略。

7. **设计评估体系**

   视觉质量、身份、口型、动作、时序、延迟、稳定性、安全合规。

<h2 id="q-032">面试问题：如何评估 GAN/Diffusion 数字人的质量？</h2>

| 维度 | 关注点 | 常见指标 |
| --- | --- | --- |
| 图像真实感 | 是否清晰自然 | FID、LPIPS、CLIP-IQA、人工评测 |
| 身份一致性 | 是否像参考人物 | ArcFace cosine similarity |
| 口型同步 | 音频和嘴部是否一致 | SyncNet/LSE-C/LSE-D |
| 表情自然度 | 情绪、眨眼、头动是否合理 | landmark dynamics、人工评分 |
| 姿态准确性 | 是否跟随驱动姿态 | keypoint error、PCK、MPJPE |
| 时序稳定性 | 是否闪烁、漂移 | optical flow consistency、temporal LPIPS |
| 背景融合 | 边界、遮挡是否自然 | mask boundary error、人工评测 |
| 推理性能 | 是否满足产品需求 | FPS、latency、显存、吞吐 |
| 安全合规 | 是否滥用身份或生成违规内容 | 水印、检测器、审核通过率 |

评估建议：

- 不要只看 FID。数字人更关心身份、口型、时序和用户主观感受。
- 对实时系统必须统计 P50/P90/P99 延迟。
- 对长视频必须评估跨段身份一致性和累积漂移。
- 对换脸必须评估遮挡、侧脸、强光、低清、人脸小目标等困难集。

<h2 id="q-033">面试问题：GAN 数字人线上常见问题如何排查？</h2>

| 问题 | 常见原因 | 排查方向 |
| --- | --- | --- |
| 身份不像 | ID embedding 弱，参考图质量差，融合权重低 | 换高质量参考图，多图 ID，增强 identity loss |
| 表情不跟随 | attribute encoder 不稳，关键点错误 | 检查 landmark/3DMM，加入表情监督 |
| 嘴型不同步 | 音频延迟、采样率错误、Sync loss 弱 | 检查音频预处理、帧率对齐、同步模型 |
| 边界有痕迹 | mask 不准，颜色不匹配 | 优化分割、Poisson/alpha blending、颜色校正 |
| 视频闪烁 | 逐帧条件抖动，无时序约束 | 条件平滑、光流约束、时序判别器 |
| 侧脸崩坏 | 训练姿态分布不足，3D 先验弱 | 增加侧脸数据，引入 3DMM/FLAME |
| 牙齿/口腔假 | 口腔区域数据不足，局部判别弱 | 嘴部 crop 训练，局部 loss，高质量口腔数据 |
| 头发/眼镜穿帮 | 遮挡 mask 错误 | occlusion-aware mask，分层融合 |
| 移动端慢 | 模型过大，后处理重 | 剪枝、量化、蒸馏、分辨率分级 |

排查顺序：

1. 先查输入：人脸检测、裁剪、音频、帧率、参考图。
2. 再查中间条件：ID embedding、landmark、mask、pose、flow。
3. 再查模型输出：生成区域、边界、嘴部、眼睛。
4. 最后查后处理和编码：超分、贴回、颜色、音视频同步。

<h2 id="q-034">面试问题：数字人安全、版权和反滥用为什么越来越重要？</h2>

数字人技术直接涉及人脸、声音、身份和视频表达，存在明显的滥用风险。

主要风险：

- 未授权换脸和声音克隆。
- 冒充真人进行诈骗、舆论操控或虚假宣传。
- 生成不当内容损害人物名誉。
- 使用未授权训练数据侵犯肖像权、版权或隐私。
- 模型输出无水印导致溯源困难。

工程和产品侧需要：

1. **授权机制**

   对身份图、声音、视频素材进行授权校验和留档。

2. **内容审核**

   对输入和输出都进行涉政、色情、暴力、诈骗、名人滥用等审核。

3. **水印与溯源**

   添加显式或隐式水印，保留生成记录。

4. **活体和真人声明**

   对高风险身份生成增加活体校验或真人授权确认。

5. **反伪造检测**

   部署 deepfake detection、音频伪造检测和异常传播监控。

6. **权限分级**

   对普通用户、企业用户、敏感行业设置不同生成权限。

面试中可以强调：数字人不是纯算法问题，产品化必须把身份授权、安全审核、水印溯源和滥用治理作为系统设计的一部分。

<h2 id="q-035">面试问题：当前 AI 数字人论文和 GitHub 项目可以如何分层跟踪？</h2>

AI 数字人方向更新很快，建议不要按“模型名字堆列表”来记，而是按任务层分层跟踪。

| 层级 | 代表方向 | 代表论文/项目 |
| --- | --- | --- |
| 实时唇形同步 | 只改嘴部或脸部局部，强调速度 | Wav2Lip、MuseTalk、LatentSync |
| 肖像动画 | 单图头像到说话/表情/头动视频 | SadTalker、AniPortrait、EMO、VASA、Hallo、Hallo2、Hallo3、LivePortrait |
| 音频驱动全身 | 音频驱动半身/全身动作和手势 | Sonic、OmniHuman、Wan-S2V、HunyuanVideo-Avatar |
| 多人对话 | 多路音频绑定多个人物 | MultiTalk、AnyTalker |
| 角色动画/替换 | 参考角色跟随驱动动作或替换视频角色 | Animate Anyone、MagicAnimate、Champ、StableAnimator、Wan-Animate |
| ID 保持和换脸 | 保持身份或替换人脸 | IP-Adapter、InstantID、PuLID、PhotoMaker、SimSwap、HiFiFace、Face-Adapter、ReFace、VividFace |
| 3D Avatar | 可视角控制、可驱动、可实时渲染 | 3DMM、FLAME、SMPL-X、NeRF Avatar、3DGS Avatar、LAM、ICo3D、AvatarPointillist |
| 实时交互系统 | ASR + LLM + TTS + Avatar 闭环 | Duix-Mobile、OpenAvatarChat、StreamAvatar、LiveKit/RTC 类集成 |

跟踪时重点看四个维度：

1. **输入条件**：图像、视频、音频、文本、姿态、3D 参数。
2. **输出范围**：嘴部、头部、半身、全身、多人、3D。
3. **模型底座**：GAN、U-Net Diffusion、DiT、Flow、NeRF、3DGS、渲染引擎。
4. **工程能力**：实时性、显存、长视频、开源程度、商业授权、可部署性。

这样可以避免被模型名称淹没，也能快速判断一个新项目应该归入哪个技术槽位。

<h2 id="q-036">面试问题：HeyGen、Synthesia、Tavus、D-ID、Akool 等产品体现了哪些落地方向？</h2>

这些产品说明 AI 数字人已经从算法 demo 进入内容生产和企业服务。

| 产品 | 典型方向 | 面试可提炼的技术/产品点 |
| --- | --- | --- |
| HeyGen | 视频口播、Avatar、翻译和本地化 | 模板化生产、多语言配音、企业视频工作流 |
| Synthesia | 企业培训和营销视频 Avatar | 稳定角色库、脚本到视频、品牌合规 |
| Tavus | Conversational Video Interface、数字孪生 | 实时对话、个性化视频、API 化 Avatar |
| D-ID | Talking avatar、Agent、实时互动 | 图像到说话人、对话式 Agent、低门槛集成 |
| Akool | 换脸、Avatar、视频营销 | 商业视频编辑、实时/批量换脸、多场景营销素材 |

产品化关注点和论文不同：

- **论文更看模型创新**：结构、损失、数据、指标。
- **产品更看稳定交付**：模板、审核、授权、延迟、成本、失败回退、工作流。
- **企业场景更看可控性**：角色一致、品牌一致、脚本可控、内容安全、多人协作。
- **实时场景更看交互体验**：首帧延迟、打断、流式 TTS、表情同步和网络传输。

面试里如果被问“数字人怎么落地”，可以从这几个产品抽象出三类商业模式：

1. **内容生产型**：脚本到视频、营销视频、课程培训。
2. **实时交互型**：客服、导购、直播、陪伴、教育。
3. **视频编辑型**：换脸、翻译、本地化、虚拟换衣、角色替换。

<h2 id="q-037">面试问题：如何建立 AI 数字人方向的持续学习清单？</h2>

建议按“基础能力、前沿模型、工程系统、产品安全”四层维护学习清单。

**1. 基础能力**

- 人脸检测、关键点、分割、头姿估计。
- 3DMM、FLAME、SMPL-X、BlendShape、LBS。
- GAN、Diffusion、DiT、Flow Matching、NeRF、3DGS。
- 音频编码器：Wav2Vec、HuBERT、Whisper、AV-HuBERT。

**2. 前沿模型**

- 每周关注 audio-driven portrait、human animation、ID-preserving generation、3D avatar、video diffusion。
- 优先看项目页、demo、代码、数据处理和推理脚本。
- 对比输入输出、显存、速度、模型大小和失败案例。

**3. 工程系统**

- ASR/LLM/TTS/Avatar 流式链路。
- WebRTC、视频编码、首帧延迟、缓存、模型量化。
- 端云协同、移动端 SDK、GPU 服务成本。

**4. 产品安全**

- 肖像权、声音授权、数字水印、内容审核。
- Deepfake 检测和生成内容溯源。
- 企业工作流中的权限、日志和审计。

学习方法：

1. 每个新模型先回答五个问题：输入是什么、输出是什么、解决了什么旧问题、依赖什么底模、能否部署。
2. 每类任务保留 3 个代表模型：经典方法、当前强方法、工程友好方法。
3. 定期用同一批测试素材评估新项目：正脸、侧脸、多人、长音频、手势、遮挡、低清。
4. 不只看 demo，也看失败案例、license、推理成本和社区 issue。

---

## 高频速记

1. GAN 数字人仍然重要，尤其在实时、局部生成、换脸、轻量部署场景。
2. StyleGAN 的核心价值是高质量人脸生成、可编辑 latent space 和 GAN inversion。
3. GAN Inversion 的核心矛盾是重建保真度和可编辑性。
4. FOMM 类方法通过关键点、运动场和遮挡图驱动肖像动画。
5. LivePortrait 说明轻量模型在实时肖像动画中仍有很强工程价值。
6. GAN 换脸的本质是源身份和目标属性解耦。
7. ArcFace identity loss 是 GAN 换脸常见身份约束。
8. 3D-aware GAN 通过相机条件和三维中间表示提升视角一致性。
9. EG3D 的 tri-plane 在质量、效率和 3D 一致性之间取得了关键平衡。
10. Diffusion/DiT 数字人更适合多条件、高保真、长视频和全身生成。
11. AIGC 数字人正在从“动嘴头像”走向“全身、多条件、长视频、可交互”。
12. 产品级数字人必须同时考虑身份授权、内容安全、水印和反滥用。
13. 追踪 AI 数字人前沿时，应按任务层、输入条件、模型底座和工程能力分层比较。
14. 商业数字人产品重点不只是生成质量，还包括授权、审核、模板工作流、API、成本和低延迟体验。

## 参考资料

- Goodfellow et al., [**Generative Adversarial Nets**](https://arxiv.org/abs/1406.2661), NeurIPS 2014.
- Karras et al., [**A Style-Based Generator Architecture for Generative Adversarial Networks**](https://arxiv.org/abs/1812.04948), CVPR 2019.
- Karras et al., [**Analyzing and Improving the Image Quality of StyleGAN**](https://arxiv.org/abs/1912.04958), CVPR 2020.
- Karras et al., [**Alias-Free Generative Adversarial Networks**](https://nvlabs.github.io/stylegan3/), NeurIPS 2021.
- Siarohin et al., [**First Order Motion Model for Image Animation**](https://aliaksandrsiarohin.github.io/first-order-model-website/), NeurIPS 2019.
- Wang et al., [**HoloGAN: Unsupervised Learning of 3D Representations From Natural Images**](https://arxiv.org/abs/1904.01326), ICCV 2019.
- Chan et al., [**pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis**](https://marcoamonteiro.github.io/pi-GAN-website/), CVPR 2021.
- Niemeyer, Geiger, [**GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields**](https://m-niemeyer.github.io/project-pages/giraffe/index.html), CVPR 2021.
- Chan et al., [**Efficient Geometry-aware 3D Generative Adversarial Networks**](https://nvlabs.github.io/eg3d/), CVPR 2022.
- Ren et al., [**LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control**](https://arxiv.org/abs/2407.03168), 2024.
- Xu et al., [**Hallo: Hierarchical Audio-Driven Visual Synthesis for Portrait Image Animation**](https://arxiv.org/abs/2406.08801), 2024.
- Cui et al., [**Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation**](https://arxiv.org/abs/2410.07718), ICLR 2025.
- Tian et al., [**EMO: Emote Portrait Alive**](https://humanaigc.github.io/emote-portrait-alive/), 2024.
- Microsoft Research, [**VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time**](https://www.microsoft.com/en-us/research/project/vasa-1/), 2024.
- ByteDance, [**OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models**](https://omnihuman-lab.github.io/), 2025.
- Kong et al., [**Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation**](https://meigen-ai.github.io/multi-talk/), 2025.
- HumanAIGC-Engineering, [**OpenAvatarChat: Open-Source AI-Powered Digital Avatars**](https://www.openavatarchat.ai/), 2025.
- Sun et al., [**StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars**](https://streamavatar.github.io/), 2025.
- HeyGen, [**AI Avatar and Video Platform**](https://www.heygen.com/).
- Synthesia, [**AI Video and Avatar Platform**](https://www.synthesia.io/).
- Tavus, [**Conversational Video Interface**](https://www.tavus.io/).
- D-ID, [**AI Agents and Talking Avatars**](https://www.d-id.com/).
- Akool, [**AI Video and Avatar Platform**](https://akool.com/).
