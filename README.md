# 多模态图像生成系统

一个基于 AI 的图像生成系统，支持文本生成图像、提示词增强优化、CLIP 语义相似度评估和 BLIP 图像描述；在 **SD 2.1** 之外，可选用 **SD 1.5 多风格** 文生图（与 SD 2.1 同一套 `/generate` 流程）。

**文档与仓库当前行为对齐说明（本版 README 更新于 2026-04）：** 启动前必须通过环境变量提供 `HF_TOKEN` 与 `DEEPSEEK_API_KEY`（推荐使用项目根目录 `.env`）。可选：`LOCAL_FILES_ONLY=true` 仅使用本地缓存；`ALLOW_HF_HUB_DOWNLOAD` 控制缓存缺失时是否允许联网从 Hugging Face 拉取（未设置时逻辑与 `LOCAL_FILES_ONLY` 联动，见下文）。

## 功能特性

- **三种生成路径**
  - **SD 2.1 基础生成** — `stable-diffusion-2-1-base`，默认 **768×768**，VAE Tiling
  - **SD 1.5 风格生成** — 在「基础生成」面板选择后端：`sd21`（默认）或 `sd15`，并指定 **五种风格之一**（二次元 / 水彩 / 油画 / 写实 / 素描），默认 **512×512**（由 `style_models.json` 中 `sd15_size` 配置）
  - **SD 1.5 + ControlNet（Canny）** — 独立流程，基于上传图 Canny 边缘控制生成（**当前版本五种 SD1.5 风格不接 ControlNet**）
- **提示词增强** — DeepSeek API 扩展中文绘画提示词
- **中译英** — 优先本地 `Helsinki-NLP/opus-mt-zh-en`；缓存不可用或加载失败时 **回退 DeepSeek API**
- **CLIP 语义相似度** — 评估生成图与提示词匹配度
- **BLIP 图像描述** — 英文描述生成图
- **语音播报** — 浏览器 Web Speech API

## 项目结构

```
├── api_server.py             # FastAPI 后端
├── index.html                # 前端单页
├── style_models.json         # SD1.5 风格：HF 仓库、权重文件名、本地回退路径、LoRA 强度与提示词提示
├── frontend/                 # React 前端（可选/备用）
├── scripts/
│   └── download_models_stepwise.py   # 分步预拉 Hugging Face 资源
├── run_app.bat               # Windows：启动服务并打开浏览器
├── run_download.bat          # Windows：预下载（默认 large，可按需改为 sd15_styles / all）
├── improve.md                # 功能与改进记录（可选，团队内可纳入版本库）
├── .env.example              # 环境变量模板
├── requirements.txt          # Python 依赖（含 diffusers、peft 等）
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 **Python 3.10+**。

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写密钥：

**Windows（cmd）：** `copy .env.example .env`  
**类 Unix：** `cp .env.example .env`

```env
HF_TOKEN=your-huggingface-token-here
DEEPSEEK_API_KEY=your-deepseek-api-key-here
# 可选：仅离线/仅缓存
# LOCAL_FILES_ONLY=true
# 可选：是否允许 Hub 下载（未设置时，LOCAL_FILES_ONLY=true 则等价于不允许）
# ALLOW_HF_HUB_DOWNLOAD=false
```

不要将真实 `.env` 提交到仓库。

### 3. 启动服务

```bash
python api_server.py
```

浏览器访问 **http://127.0.0.1:8000**。**Windows** 可使用 `run_app.bat`（自动查找上级 `.venv-gpu`、本项目 `.venv310` 或 `PATH` 中的 `python`）。

## 模型预下载（可选）

脚本 **`scripts/download_models_stepwise.py`** 按仓库或单文件拉取资源，并为子进程设置较长的 Hub 超时（`HF_HUB_ETAG_TIMEOUT` / `HF_HUB_DOWNLOAD_TIMEOUT`）。

**分组说明：**

| `--group` | 内容 |
|-----------|------|
| `medium` | 翻译、CLIP、BLIP |
| `large` | SD2.1、SD1.5 底模、ControlNet-Canny |
| `sd15_styles` | DreamShaper、油画 LoRA 仓库 **snapshot**；另对 Counterfeit、水彩、素描等使用 **`hf_hub_download` 单文件**，避免整库过大或 Windows 下过长路径问题 |
| `all` | 上述全部 |

**Windows：** `run_download.bat` 当前默认为 `--group large`。若需 SD1.5 风格资源，请将 bat 中参数改为 `sd15_styles` 或 `all`。

示例：

```bash
python scripts/download_models_stepwise.py --python "C:\path\to\python.exe" --group sd15_styles --timeout 7200
```

权重与本地路径见 **`style_models.json`**（`local_fallback`）。若 HF 失败，可将同名文件放到对应路径（需自行获取权重）。

## 环境变量与联网策略（摘要）

| 变量 | 含义 |
|------|------|
| `LOCAL_FILES_ONLY=true` | 加载模型时 `local_files_only=True`，不主动从 Hub 拉取 |
| `ALLOW_HF_HUB_DOWNLOAD` | 未设置时：**若未** `LOCAL_FILES_ONLY`，则允许在缓存缺失时联网；若 `LOCAL_FILES_ONLY=true`，则不允许。也可显式设为 `true` / `false` 覆盖 |

## 技术栈

- FastAPI、Uvicorn  
- PyTorch、**Diffusers（≥0.27）**、Transformers、**PEFT**（SD1.5 LoRA）  
- OpenCV、Pillow、python-dotenv  

## 模型与风格说明（摘要）

- **SD 2.1：** `sd2-community/stable-diffusion-2-1-base`  
- **ControlNet：** `runwayml/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-canny`  
- **SD 1.5 风格（`generation_mode=sd15`）：** 由 `style_models.json` 配置，例如二次元用 `gsdf/Counterfeit-V3.0` 单文件权重、写实基于 `Lykon/dreamshaper-8`，水彩/油画/素描为 DreamShaper + 对应 LoRA（详见该 JSON）  
- **CLIP：** `openai/clip-vit-base-patch32`  
- **BLIP：** `Salesforce/blip-image-captioning-large`  
- **翻译：** `Helsinki-NLP/opus-mt-zh-en`（可选 DeepSeek 回退）

## 安装部署（给他人使用）

1. 克隆后进入项目根目录，创建虚拟环境并 `pip install -r requirements.txt`。  
2. 复制 `.env.example` 为 `.env` 并填写密钥。  
3. 预留足够磁盘：**基础套件约 15GB+**；若使用 **SD1.5 风格**，额外需要 Counterfeit / DreamShaper / LoRA 等空间。  
4. 可将手动下载的权重放到 `style_models.json` 的 `local_fallback` 路径；**`models/` 目录已在 `.gitignore` 中忽略**，避免误提交大文件。  
5. 有 NVIDIA GPU + CUDA 时推理更快。

## 工作流程

```
用户输入 →（可选）扩充提示词 → 中译英 → SD 2.1 或 SD1.5 风格 或（另一入口）ControlNet
         → BLIP + CLIP →（可选）语音播报
```

## API 接口摘要

### `GET /status`

返回 `device`、`cuda_available`、`models_loaded`（含 `sd21`、`sd15_anime`、`sd15_dreamshaper`、`blip`、`clip`、`controlnet` 等是否已加载）。

### `POST /generate`

| 字段 | 说明 |
|------|------|
| `prompt` | 用户提示（可中文） |
| `enhanced_prompt` | 可选；若提供则优先生效并参与翻译 |
| `num_inference_steps`、`guidance_scale` | 与惯例相同 |
| `generation_mode` | `"sd21"`（默认）或 `"sd15"` |
| `sd15_style` | 当 `generation_mode` 为 `sd15` 时 **必填**：`anime` \| `watercolor` \| `oil` \| `realistic` \| `sketch` |

成功时响应额外包含 `generation_mode`、`sd15_style`（仅 sd15 时有值）。

### 其他

- `POST /enhance`、`POST /process-canny`、`POST /generate-controlnet`、`POST /clip-evaluate`、`POST /caption` 行为见前文版本说明；ControlNet 请求体中 `control_image` 为 **纯 base64**（与 `index.html` 中 Canny 结果 `split(',')[1]` 一致）。

## 前端使用

1. 打开服务根 URL。  
2. **SD 2.1 基础生成**：可选择生成后端为 **SD 2.1** 或 **SD 1.5** 及五种风格。  
3. **SD 1.5 + ControlNet**：上传参考图 → Canny → 生成（与上述五种风格入口分离）。  
4. 可使用「扩充提示词」、查看 CLIP 与 BLIP、语音播报。

## CLIP 分数参考

| 相似度（约） | 含义 |
| ------------ | ---- |
| > 0.80       | 很高匹配 |
| 0.60–0.80    | 较好 |
| 0.40–0.60    | 一般 |
| < 0.40       | 偏低 |

*CLIP 输出为约 0～1 的余弦相似度，表内为经验区间。*

## 常见问题

### 启动报错缺少 Token？

检查 `.env` 中 `HF_TOKEN`、`DEEPSEEK_API_KEY`。

### SD1.5 风格加载失败？

先运行 `download_models_stepwise.py --group sd15_styles`（或 `all`），或按 `style_models.json` 手动放置 `local_fallback` 文件；确认 `ALLOW_HF_HUB_DOWNLOAD` / `LOCAL_FILES_ONLY` 与网络环境一致。

### `run_download.bat` 没有拉 SD1.5 风格？

默认仅 `large`。请改为 `--group sd15_styles` 或 `all`。

### 翻译不走本地模型？

缓存中无 Helsinki 或加载失败时会使用 DeepSeek 做中译英（需有效 `DEEPSEEK_API_KEY`）。

## 环境要求

- Python 3.10+  
- PyTorch 2.x  
- 内存建议 8GB+；磁盘建议 **15GB+**（含 SD1.5 风格时更多）  
- 推荐 NVIDIA GPU，显存 8GB+ 更稳妥  

## License

MIT License
