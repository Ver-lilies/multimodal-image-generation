# 多模态图像生成系统

一个基于AI的图像生成系统，支持文本生成图像、提示词增强优化和智能参数调整。

## 功能特性

- 🎨 **文本生成图像** - 使用 Stable Diffusion 2.1 将文本描述转化为图像
- ✨ **提示词增强** - 使用 DeepSeek API 优化提示词
- 🔍 **CLIP 提示词评估** - 使用 CLIP 模型评估提示词质量
- 🤖 **智能参数优化** - 根据 CLIP 评分自动调整推理步数和引导系数
- 🌐 **多语言支持** - 支持中文提示词自动翻译为英文
- 🔊 **语音播报** - 自动朗读生成的图像描述

## 项目结构

```
├── api_server.py             # FastAPI 后端服务（主入口）
├── image_generator.py        # Stable Diffusion 图像生成模块
├── image_captioner.py        # BLIP 图像描述生成模块
├── clip_prompt_alignment.py  # CLIP 提示词评估模块
├── tts_player.py             # TTS 语音播放模块
├── index.html                # 前端页面
├── frontend/                 # React 前端项目
├── requirements.txt          # Python 依赖
└── README.md                 # 项目说明文档
```

## 技术栈

### 后端

- **FastAPI** - Web 框架
- **PyTorch** - 深度学习框架
- **Diffusers** - Stable Diffusion 模型加载
- **Transformers** - BLIP、CLIP、翻译模型

### 前端

- **HTML/CSS/JavaScript** - 原生前端

### API 服务

- **DeepSeek API** - 提示词增强

## 模型说明

### 1. Stable Diffusion 2.1 (图像生成)

- **模型ID**: `sd2-community/stable-diffusion-2-1-base`
- **用途**: 根据文本提示生成图像
- **默认参数**:
  - 推理步数: 50
  - 引导系数: 10.0
  - 分辨率: 768x768

### 2. CLIP (提示词评估)

- **模型ID**: `openai/clip-vit-base-patch32`
- **用途**: 评估提示词质量，自动优化生成参数
- **功能**:
  - 提示词质量评分（基于CLIP相似度）
  - 扩展提示词与原始提示词的语义相似度
  - 智能参数优化（低质量时自动增加步数和系数）

### 3. BLIP (图像描述)

- **模型ID**: `Salesforce/blip-image-captioning-base`
- **用途**: 生成图像的文字描述

### 4. 翻译模型 (中文→英文)

- **模型ID**: `Helsinki-NLP/opus-mt-zh-en`
- **用途**: 将中文提示词翻译为英文

## 安装部署

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API Key

在 `api_server.py` 中配置以下 API Key：

```python
HF_TOKEN = 'your-huggingface-token'      # HuggingFace Token
DEEPSEEK_API_KEY = 'your-deepseek-key'    # DeepSeek API Key
```

### 4. 首次运行

首次运行时会自动从 HuggingFace 下载模型到本地缓存（约 6-8GB）：

- Stable Diffusion 2.1 Base (\~4GB)
- BLIP Image Captioning Base (\~400MB)
- Opus MT 中文→英文 翻译模型 (\~300MB)
- CLIP 提示词评估模型 (\~500MB)

### 5. 启动服务

```bash
python api_server.py
```

服务启动后访问: **<http://localhost:3000>**

## 工作流程

```
用户输入提示词
      ↓
  ↓ 扩充提示词 (DeepSeek API)
      ↓
CLIP 评估扩展后的提示词
      ↓
  ↓ 自动优化参数（根据质量评分）
      ↓     
Stable Diffusion 生成图像
      ↓
  ↓ BLIP 生成图像描述
      ↓
语音播报描述
```

## API 接口

### 1. 图像生成

```bash
POST /generate
Content-Type: application/json

{
    "prompt": "一只可爱的猫咪",
    "enhanced_prompt": "扩展后的提示词",
    "num_inference_steps": 50,
    "guidance_scale": 10.0,
    "clip_evaluation_score": 0.51,
    "use_clip_optimization": true
}
```

返回结果：

```json
{
    "image": "data:image/png;base64,...",
    "original_prompt": "一只可爱的猫咪",
    "translated_prompt": "A lovely cat...",
    "caption": "a cat sitting on...",
    "clip_evaluation": {
        "quality_score": 0.51,
        "similarity_score": 0.47,
        "word_count": 27,
        "has_details": true
    },
    "used_num_steps": 60,
    "used_guidance": 11.5
}
```

### 2. 提示词增强

```bash
POST /enhance
Content-Type: application/json

{
    "prompt": "一只猫"
}
```

### 3. CLIP 提示词评估

```bash
POST /clip-evaluate
Content-Type: application/json

{
    "prompt": "A lovely cat sitting on window..."
}
```

### 4. 状态检查

```bash
GET /status
```

## 前端使用

1. 打开 `http://localhost:3000`
2. 在文本框中输入描述（如"一只可爱的猫咪"）
3. 点击"✨ 扩充提示词"获得更好的效果
4. 点击"🔍 CLIP提示词评估"评估扩展后的提示词
5. 点击"🚀 生成图像"
6. 系统会自动根据CLIP评分优化参数并生成图像

## CLIP 智能优化说明

| 质量评分    | 参数调整              |
| ------- | ----------------- |
| < 50%   | 推理步数+20, 引导系数+3   |
| 50%-70% | 推理步数+10, 引导系数+1.5 |
| > 70%   | 保持原参数             |

## 常见问题

### Q: 生成图像有噪点怎么办？

A: 增加推理步数，或让 CLIP 自动优化（质量评分低时会自动增加）

### Q: 图像不符合预期怎么办？

A:

1. 使用"扩充提示词"功能
2. 多次调整直到 CLIP 评分较高

### Q: 模型加载失败怎么办？

A:

1. 检查网络连接
2. 确认 HuggingFace Token 正确
3. 确保有足够的磁盘空间

### Q: 生成速度慢怎么办？

A:

1. 使用 GPU 加速（需要 CUDA）
2. 减少推理步数

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM
- 10GB+ 磁盘空间
- 推荐: NVIDIA GPU with 6GB+ VRAM

## License

MIT License
