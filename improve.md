# 项目改进记录

本文档记录本仓库后续的功能与质量改进，按时间倒序追加。

---

## 2026-04-17 — SD 1.5 五种风格文生图（与 SD 2.1 同流程）

### 功能

- 在 **「SD 2.1 基础生成」** 面板增加 **生成后端** 选择：`SD 2.1`（默认）或五种 **SD 1.5 风格**（二次元 / 水彩 / 油画 / 写实 / 素描）。
- **POST `/generate`** 增加字段：`generation_mode`（`sd21` | `sd15`）、`sd15_style`（仅 `sd15` 时必填）。
- SD 1.5 分支分辨率固定为 **512×512**（第一版）；流程仍为：翻译 → 扩散 → BLIP → CLIP。
- **SD 1.5 + ControlNet** 未改动，五种风格第一版 **不接 ControlNet**。

### 模型与资源（Hugging Face）

| 风格 | 说明 |
|------|------|
| 二次元 | `gsdf/Counterfeit-V3.0` → `Counterfeit-V3.0_fp16.safetensors`（`from_single_file`） |
| 写实 | `Lykon/dreamshaper-8`（无 LoRA） |
| 水彩 | DreamShaper + `fladdict/watercolor` → `fladdict-watercolor-sd-1-5.safetensors` |
| 油画 | DreamShaper + `jqlive/sd15-digital-oil-arcane`（偏数码/Arcane 油画风；可本地替换 LoRA） |
| 素描 | DreamShaper + `jordanhilado/sd-1-5-sketch-lora` → 根目录 `pytorch_lora_weights.safetensors` |

权重与本地回退路径见 **`style_models.json`**。若 HF 下载失败，可将同名文件放到 `local_fallback` 所示路径（需自行从 HF 网页或镜像下载）。

### 下载

- `scripts/download_models_stepwise.py` 增加 `--group sd15_styles`（或 `--group all` 会包含这些资源），用于预拉取 HF 仓库。
- **Counterfeit**：仅 `hf_hub_download` `Counterfeit-V3.0_fp16.safetensors`，不对 `gsdf/Counterfeit-V3.0` 做整库 snapshot（避免拉取约 9GB 全量变体）。
- **水彩 LoRA**：仅下载 `fladdict-watercolor-sd-1-5.safetensors`，不对 `fladdict/watercolor` 做整库 snapshot（仓库内预览图文件名过长，Windows 下整库 snapshot 易触发 symlink/`FileNotFoundError`）。
- **素描 LoRA**：仅 `hf_hub_download` 根目录 `pytorch_lora_weights.safetensors`，避免整库 snapshot。
- 示例（按你本机 Python 路径修改）：

```bat
python scripts\download_models_stepwise.py --python "E:\Course\Statistic_L\.venv-gpu\Scripts\python.exe" --group sd15_styles --timeout 7200
```

### 手动下载（脚本失败或需离线时）

1. 登录 [Hugging Face](https://huggingface.co)，在网页上打开对应模型卡，下载与 `style_models.json` 中一致的文件名，放到 HF 默认缓存目录，或放到 `local_fallback` 指向的相对路径（相对于项目根目录）。
2. **Counterfeit**：至少下载 `Counterfeit-V3.0_fp16.safetensors` 到 `models/checkpoints/`（或配置的路径）。
3. **素描 LoRA**：只下载 `jordanhilado/sd-1-5-sketch-lora` 仓库根目录的 `pytorch_lora_weights.safetensors` 即可，勿拉取全仓库。

### 依赖

- 建议 `diffusers>=0.27.0` 以稳定支持 `load_lora_weights` / `unload_lora_weights`。
- **`peft`**：`load_lora_weights` 需要 PEFT 后端；已加入 `requirements.txt`（若报错 `PEFT backend is required for this method` 请执行 `pip install peft`）。
- **素描 LoRA**：`jordanhilado/sd-1-5-sketch-lora` 的权重键为自定义 `processor.*_lora` 形式，Diffusers 无法映射到 `UNet2DConditionModel`（日志中 “No LoRA keys associated… prefix='unet'”），已改为 `davidberenstein1957/p-image-pencil-sketch-art-lora` / `weights.safetensors`；英文提示可自动加前缀 `A color pencil sketch of`。

---

（后续改进请在本节上方继续追加。）
