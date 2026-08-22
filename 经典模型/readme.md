# 《三年面试五年模拟》之经典模型知识高频考点

本板块面向 AIGC 算法岗、计算机视觉算法岗、多模态算法岗、NLP / LLM 算法岗、AI 工程岗面试，系统梳理跨周期经典模型，并补充 AIGC 时代常用、常考、必备的基础模型知识。

经典模型不是过时模型，而是 AIGC 模型的结构基础、任务基础和工程基础。CNN、Transformer、检测、分割、OCR、人脸、ReID、强化学习、迁移学习、多模态对齐等知识，仍然贯穿今天的视觉基础模型、扩散模型、视频生成、文档智能、多模态 Agent 和大模型对齐系统。

## 推荐阅读顺序

1. [图像分类高频知识点](图像分类高频知识点.md)

   重点掌握 CNN 骨干、轻量化模型、ViT/Swin/ConvNeXt、MAE、DINOv2、CLIP、DiT 等视觉 backbone 和 AIGC 时代视觉预训练范式。

2. [目标检测高频知识点](目标检测高频知识点.md)

   重点掌握 R-CNN 系列、YOLO、SSD、RetinaNet、FPN、DETR、开放词汇检测、Grounding DINO、YOLO-World、RT-DETR、YOLOv10/YOLO11 等检测主线。

3. [图像分割高频知识点](图像分割高频知识点.md)

   重点掌握 FCN、U-Net、DeepLab、Mask R-CNN、Mask2Former、SAM、SAM 2、SEEM，以及分割在 AIGC 图像编辑、视频编辑、自动标注中的应用。

4. [自然语言处理高频知识点](自然语言处理高频知识点.md)

   重点掌握词向量、Seq2Seq、Attention、Transformer、BERT、GPT、T5、CLIP、BLIP、Flamingo、LLaVA、Florence-2，以及 NLP 与 RAG、多模态 Agent 的关系。

5. [强化学习高频知识点](强化学习高频知识点.md)

   重点掌握 MDP、Q-Learning、DQN、Policy Gradient、Actor-Critic、PPO、RLHF、DPO、RLAIF、LoRA、Adapter、持续学习和模型对齐。

6. 专项补充：

   - [OCR高频知识点](OCR高频知识点.md)
   - [人脸模型高频知识点](人脸模型高频知识点.md)
   - [目标跟踪高频知识点](目标跟踪高频知识点.md)
   - [ReID高频知识点](ReID高频知识点.md)
   - [VAE与视觉潜空间高频知识点](VAE与视觉潜空间高频知识点.md)

## 内容边界

| 文档 | 核心职责 | 主要关注点 |
| --- | --- | --- |
| 图像分类 | 视觉 backbone 与分类模型 | CNN、轻量化、ViT、自监督、DINO、DiT |
| 目标检测 | 目标定位与开放词汇检测 | Two-stage、One-stage、DETR、YOLO、OVD |
| 图像分割 | 像素级理解与视觉基础模型 | FCN、U-Net、DeepLab、Mask R-CNN、SAM |
| VAE 与视觉潜空间 | 图像压缩、重构与生成潜变量 | AE、VAE、VQ-VAE、重参数化、潜空间扩散 |
| 自然语言处理 | NLP、多模态与生成基础模型 | Transformer、BERT/GPT、VLM、RAG、Agent |
| 强化学习 | 决策优化、迁移学习与对齐 | RL、RLHF、DPO、LoRA、持续学习 |
| OCR / 人脸 / 跟踪 / ReID | 视觉专项模型 | 文档智能、身份识别、视频理解 |

人脸章节覆盖从检测、关键点与对齐，到 embedding、验证/识别/检索、Face Parsing、属性分析、美颜、活体与 DeepFake 检测，以及 AIGC 身份保持的完整链路。人脸生成产品的业务选型、成本和合规运营放在 [AI业务场景问题](../AI行业职业生涯开放性问题/AI业务场景问题.md) 中，本章只保留可迁移的模型机制、工程接口和评估方法。

## 跨周期核心知识检查清单

- CNN 经典骨干：LeNet、AlexNet、VGG、GoogLeNet/Inception、ResNet、DenseNet、MobileNet、EfficientNet、RepVGG。
- 视觉 Transformer 与自监督：ViT、Swin、ConvNeXt、MAE、DINOv2、CLIP、DiT。
- 目标检测：R-CNN、Fast R-CNN、Faster R-CNN、FPN、SSD、YOLO、RetinaNet、FCOS、CenterNet、DETR、Deformable DETR、DINO、RT-DETR、Grounding DINO、YOLO-World。
- 图像分割：FCN、U-Net、SegNet、PSPNet、DeepLab、Mask R-CNN、Mask2Former、SAM、SAM 2、SEEM。
- NLP 与多模态：Word2Vec、Seq2Seq、Attention、Transformer、BERT、GPT、T5、CLIP、BLIP、Flamingo、LLaVA、Florence-2。
- 强化学习与模型对齐：MDP、Q-Learning、DQN、Policy Gradient、Actor-Critic、PPO、RLHF、DPO、RLAIF。
- 迁移学习与 PEFT：Domain Adaptation、Fine-tuning、负迁移、LoRA、Adapter、Prefix Tuning、Prompt Tuning、持续学习。
