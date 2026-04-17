# 多模态图像生成系统

一个基于 AI 的图像生成系统，支持文本生成图像、提示词增强优化、CLIP 语义相似度评估和 BLIP 图像描述。

**文档与仓库当前行为对齐说明（本版 README 更新于 2026-04）：** 启动前必须通过环境变量提供 `HF_TOKEN` 与 `DEEPSEEK_API_KEY`（推荐使用项目根目录 `.env`）；可选 `LOCAL_FILES_ONLY=true` 在模型已缓存时仅使用本地文件、避免联网拉取。

## 功能特性

- **双模式图像生成**
  - **SD 2.1 基础生成** — 使用 Stable Diffusion 2.1 将文本描述转化为图像（默认 768×768，VAE Tiling）
  - **SD 1.5 + ControlNet** — 使用 ControlNet（Canny）基于边缘图控制生成
- **提示词增强** — 使用 DeepSeek API 将用户输入扩展为简洁中文绘画提示词
- **CLIP 语义相似度** — 评估生成图与提示词的匹配程度
- **BLIP 图像描述** — 为生成图自动生成英文描述
- **多语言** — 中文提示词经翻译模型转为英文后喂给 SD/CLIP
- **语音播报** — 浏览器 Web Speech API 朗读描述（无需后端 TTS）

## 项目结构

```
├── api_server.py             # FastAPI 后端（集成生成、评估、静态首页）
├── index.html                # 前端单页（与 API 同进程同端口）
├── frontend/                 # React 前端（可选/备用）
├── scripts/
│   └── download_models_stepwise.py   # 按步骤从 Hugging Face 预拉模型快照
├── run_app.bat               # Windows：选择 Python 后启动服务并打开浏览器
├── run_download.bat          # Windows：调用分步下载脚本（见下文）
├── .env.example              # 环境变量模板（复制为 .env）
├── requirements.txt          # Python 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 **Python 3.10+**（代码中使用 `str | None` 等类型写法）。

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写密钥：

**Windows（cmd）：**

```bat
copy .env.example .env
```

**类 Unix：**

```bash
cp .env.example .env
```

`.env` 示例字段：

```env
HF_TOKEN=your-huggingface-token-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
# 可选：仅使用已缓存的模型，不发起下载（需本地已有缓存）
# LOCAL_FILES_ONLY=true
```

不要将真实 `.env` 提交到仓库。

### 3. 启动服务

**任意系统：**

```bash
python api_server.py
```

浏览器访问 **http://127.0.0.1:8000** 或 **http://localhost:8000**（首页由 `GET /` 返回 `index.html`）。

**Windows 便捷方式：** 双击或在资源管理器中运行 `run_app.bat`。脚本会依次尝试上级目录的 `.venv-gpu`、本项目下的 `.venv310`、以及 `PATH` 中的 `python`，然后新开窗口启动 `api_server.py`，并尝试打开默认浏览器。

## 模型预下载（可选）

首次推理时，`diffusers` / `transformers` 仍可能按需下载权重。若在弱网或希望先统一下载，可使用 **`scripts/download_models_stepwise.py`**，按模型逐个调用 `huggingface_hub.snapshot_download`，并为每个仓库单独设置超时。

**Windows：** 运行 `run_download.bat`（内部调用上述脚本，`--group large` 与较长超时，可按需改 bat 内参数）。

**命令行示例：**

```bash
python scripts/download_models_stepwise.py --python "C:\path\to\python.exe" --group medium --timeout 600
```

- `--group`：`medium`（翻译、CLIP、BLIP）、`large`（SD2.1、SD1.5、ControlNet-Canny）、`all`
- 脚本会加载项目根目录 `.env` 以读取 `HF_TOKEN`

## 技术栈

### 后端

- FastAPI、Uvicorn
- PyTorch、Diffusers、Transformers
- OpenCV（Canny）、Pillow、python-dotenv

### 前端

- 原生 HTML / CSS / JavaScript（主入口由后端托管）

### 外部 API

- DeepSeek Chat API（提示词增强）
- Hugging Face Hub（模型与 Token）

## 模型说明

### Stable Diffusion 2.1（基础生成）

- **模型 ID：** `sd2-community/stable-diffusion-2-1-base`
- **说明：** 文本条件生成；启用 VAE Tiling 减轻拼缝感

### Stable Diffusion 1.5 + ControlNet Canny

- **流水线：** `runwayml/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-canny`
- **说明：** 需先由上传图得到 Canny 图（见 `/process-canny`）

### CLIP

- **模型 ID：** `openai/clip-vit-base-patch32`
- **说明：** 余弦相似度评估提示词与图像的语义一致性

### BLIP

- **模型 ID：** `Salesforce/blip-image-captioning-large`

### 翻译（中→英）

- **模型 ID：** `Helsinki-NLP/opus-mt-zh-en`

### 语音播报

- 浏览器 `SpeechSynthesis` API，无服务端语音模型依赖

## 安装部署（给他人使用）

1. **克隆仓库**后进入项目根目录（含 `api_server.py` 的目录）。
2. **创建虚拟环境**（推荐），再 `pip install -r requirements.txt`。
3. **仅通过 `.env` 配置密钥**（不要要求在源码里改 Key；若文档仍写「改 `api_server.py`」请以本 README 为准）。
4. **磁盘与网络：** 全部模型与缓存体积较大，请预留约 **15GB+** 空间；需要能访问 Hugging Face（或事先在本机缓存完毕并视情况设置 `LOCAL_FILES_ONLY`）。
5. **GPU：** 有 NVIDIA GPU + CUDA 时推理显著快于 CPU。

## 工作流程

```
用户输入提示词
      ↓
提示词增强（DeepSeek，可选）
      ↓
中译英（Opus-MT）
      ↓
Stable Diffusion 生成图像
      ↓
CLIP 相似度 + BLIP 描述（可选朗读）
```

## API 接口摘要

### `GET /`

返回单页 `index.html`；若无文件则返回 JSON 状态。

### `GET /status`

返回运行环境与模型是否已加载等信息（如 `device`、`cuda_available`、`models_loaded`）。

### `POST /enhance`

请求体：`{"prompt": "..."}`  
返回：`original_prompt`、`enhanced_prompt`

### `POST /generate`

请求体示例：

```json
{
  "prompt": "一只可爱的猫咪",
  "enhanced_prompt": "可选：若提供则优先生效并参与翻译",
  "num_inference_steps": 50,
  "guidance_scale": 10.0
}
```

返回含：`image`（data URL）、`translated_prompt`、`caption`、`clip_evaluation`、`used_num_steps`、`used_guidance`

### `POST /generate-controlnet`

请求体含：`prompt`、`control_image`（**纯 base64 字符串**，不含 `data:image/...;base64,` 前缀；与当前 `index.html` 中由 Canny 结果拆分后的字段一致）、`num_inference_steps`、`guidance_scale`、`controlnet_conditioning_scale`，以及可选 `enhanced_prompt`

### `POST /process-canny`

`multipart/form-data` 上传文件字段 `file`，返回 Canny 图的 data URL 等

### `POST /clip-evaluate`

请求体：`prompt`、`image`（base64，需有效图像以计算相似度）

### `POST /caption`

`multipart/form-data` 上传图像，`file` 字段；返回 `caption` 文本

## 前端使用

1. 打开服务根 URL（默认 8000 端口）
2. 选择 **SD 2.1** 或 **ControlNet** 流程，按页面提示上传/输入
3. 可使用「扩充提示词」再生成  
4. 生成后可查看 CLIP 分数与 BLIP 描述，并可使用语音播报

## CLIP 分数参考

| 相似度（约） | 含义 |
| ------------ | ---- |
| > 0.80       | 很高匹配 |
| 0.60–0.80    | 较好 |
| 0.40–0.60    | 一般 |
| < 0.40       | 偏低（可与提示词、随机性有关） |

*注：实际数值范围为模型输出，表内为经验区间。*

## 常见问题

### 生成图有接缝或伪影？

已启用 VAE Tiling；仍有问题时可适当增加步数或调整提示词。

### 启动报错「HF_TOKEN / DEEPSEEK_API_KEY 未设置」？

检查项目目录下是否存在正确配置的 `.env`，或是否在系统环境中导出同名变量。

### Windows 下控制台乱码或 emoji 报错？

服务端已对 stdout/stderr 尝试设为 UTF-8；若仍有终端兼容问题，可使用英文日志或减少特殊字符。

### 模型下载慢或失败？

检查网络与 `HF_TOKEN`；可先运行 `download_models_stepwise.py` 分段下载；或使用已缓存的机器并考虑 `LOCAL_FILES_ONLY=true`。

### 只想离线已有缓存？

在确认 Hugging Face 缓存目录中已有对应模型文件后，设置 `LOCAL_FILES_ONLY=true`，避免程序再次发起下载。

## 环境要求

- **Python 3.10+**
- **PyTorch 2.x**（根据平台安装 CPU/GPU 轮次）
- **内存** 建议 8GB+  
- **磁盘** 建议预留 15GB+ 用于模型与缓存  
- **GPU（推荐）** NVIDIA GPU，显存 8GB+ 更稳妥

## License

MIT License
