import os
import sys
import json
import gc
import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import Response
from pydantic import BaseModel
import uvicorn
import threading

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Windows 默认控制台常为 GBK，emoji 会导致 UnicodeEncodeError，进程在绑定端口前退出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HF_TOKEN = os.getenv('HF_TOKEN', '')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
LOCAL_FILES_ONLY = os.getenv('LOCAL_FILES_ONLY', 'false').lower() == 'true'
# Unset: same as before — allow Hub when LOCAL_FILES_ONLY is false. Set to false to use cache-only (fast fail if Hub is blocked).
_allow_raw = os.getenv("ALLOW_HF_HUB_DOWNLOAD")
ALLOW_HF_HUB_DOWNLOAD = (not LOCAL_FILES_ONLY) if _allow_raw is None else (_allow_raw.lower() == "true")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN 环境变量未设置！请在 .env 文件中配置或设置环境变量")
if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY 环境变量未设置！请在 .env 文件中配置或设置环境变量")

print(f"🔐 API Keys loaded: HF_TOKEN={'✓' if HF_TOKEN else '✗'}, DEEPSEEK={'✓' if DEEPSEEK_API_KEY else '✗'}")
print(f"📁 LOCAL_FILES_ONLY: {LOCAL_FILES_ONLY}")
print(f"🌐 ALLOW_HF_HUB_DOWNLOAD: {ALLOW_HF_HUB_DOWNLOAD}")

app = FastAPI(title="Multimodal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_generator = None
captioner = None
translator = None
clip_alignment = None
controlnet = None

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_PROJECT_DIR, "style_models.json"), encoding="utf-8") as _style_file:
    STYLE_CONFIG = json.load(_style_file)

sd15_anime_pipe = None
sd15_dreamshaper_pipe = None
sd15_dreamshaper_lora_style = None


def _hf_local_first(load, desc: str):
    """Prefer HF disk cache only; optional Hub access only when ALLOW_HF_HUB_DOWNLOAD.
    `load` must be a one-arg callable: bool -> model (e.g. lambda lfo: from_pretrained(..., local_files_only=lfo)).
    """
    try:
        return load(True)
    except Exception as e:
        if LOCAL_FILES_ONLY or not ALLOW_HF_HUB_DOWNLOAD:
            raise RuntimeError(
                f"{desc}: not in Hugging Face cache. Pre-download the model, or set "
                "ALLOW_HF_HUB_DOWNLOAD=true (requires a working connection to huggingface.co)."
            ) from e
        return load(False)


class GenerateRequest(BaseModel):
    prompt: str
    enhanced_prompt: str | None = None
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    controlnet_image: str | None = None
    generation_mode: str = "sd21"
    sd15_style: str | None = None

class ControlNetGenerateRequest(BaseModel):
    prompt: str
    control_image: str  # base64 encoded image
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    controlnet_conditioning_scale: float = 1.0
    enhanced_prompt: str | None = None

class EnhanceRequest(BaseModel):
    prompt: str

class CLIPEvaluateRequest(BaseModel):
    prompt: str
    image: str | None = None

def enhance_prompt(text: str) -> str:
    import requests
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """你是一个专业的AI绘画提示词生成器。请将用户输入扩展成简洁的中文图像生成提示词。

重要规则（必须遵守）：
1. 输出控制在50-80字左右，不要太长
2. 只输出中文提示词
3. 核心内容放前面：主体 + 风格 + 光线 + 氛围
4. 用逗号或顿号分隔关键词
5. 不要写完整句子
6. 包含：主体描述、艺术风格、光线、氛围、构图"""

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请将以下文本扩展成简洁的中文AI绘画提示词：{text}"}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }
    
    response = requests.post(url, headers=headers, json=data)
    result = response.json()
    
    enhanced = result['choices'][0]['message']['content'].strip()
    print(f"Enhanced: '{text}' -> '{enhanced}'")
    return enhanced

def _translate_zh_to_en_deepseek(text: str) -> str:
    import requests
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You translate Chinese to English for image generation prompts. Output only the English translation, no quotes or explanation.",
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    out = response.json()["choices"][0]["message"]["content"].strip()
    return out

def get_translator():
    global translator
    if translator is not None:
        return translator
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = "Helsinki-NLP/opus-mt-zh-en"
    print("Loading Translator (zh->en)...")
    tokenizer, model = None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
    except Exception:
        if not LOCAL_FILES_ONLY and ALLOW_HF_HUB_DOWNLOAD:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, local_files_only=False)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=HF_TOKEN, local_files_only=False)
            except Exception as ex:
                print(f"Helsinki model unavailable ({ex!r}); using DeepSeek API for zh->en.")
                translator = {"mode": "deepseek"}
                return translator
        else:
            print("Helsinki model not in local cache; using DeepSeek API for zh->en.")
            translator = {"mode": "deepseek"}
            return translator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    translator = {"tokenizer": tokenizer, "model": model, "device": device, "mode": "local"}
    print(f"Translator loaded on {device}!")
    return translator

def translate_to_english(text: str) -> str:
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if not has_chinese:
        return text
    trans = get_translator()
    if trans.get("mode") == "deepseek":
        translated = _translate_zh_to_en_deepseek(text)
        print(f"Translated (DeepSeek): '{text}' -> '{translated}'")
        return translated
    inputs = trans["tokenizer"](text, return_tensors="pt", padding=True).to(trans["device"])
    with torch.no_grad():
        outputs = trans["model"].generate(**inputs, max_length=512)
    translated = trans["tokenizer"].decode(outputs[0], skip_special_tokens=True)
    print(f"Translated: '{text}' -> '{translated}'")
    return translated

def get_generator():
    global image_generator
    if image_generator is None:
        unload_all_sd15()
        import logging
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        import sys
        from io import StringIO
        
        print("Loading Stable Diffusion 2.1...")
        image_generator = _hf_local_first(
            lambda lfo: StableDiffusionPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2-1-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                token=HF_TOKEN,
                local_files_only=lfo,
            ),
            "sd2-community/stable-diffusion-2-1-base",
        )
        image_generator.scheduler = DDIMScheduler.from_config(image_generator.scheduler.config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_generator = image_generator.to(device)
        print(f"SD loaded on {device}!")
    return image_generator

def get_captioner():
    global captioner
    if captioner is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("Loading BLIP Large...")
        processor = _hf_local_first(
            lambda lfo: BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-large", token=HF_TOKEN, local_files_only=lfo
            ),
            "Salesforce/blip-image-captioning-large (processor)",
        )
        model = _hf_local_first(
            lambda lfo: BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", token=HF_TOKEN, local_files_only=lfo
            ),
            "Salesforce/blip-image-captioning-large (model)",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        captioner = {"processor": processor, "model": model}
        print(f"BLIP loaded on {device}!")
    return captioner

def get_clip():
    global clip_alignment
    if clip_alignment is None:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP for image-text similarity...")
        clip_model = _hf_local_first(
            lambda lfo: CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=lfo
            ),
            "openai/clip-vit-base-patch32 (model)",
        )
        processor = _hf_local_first(
            lambda lfo: CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=lfo
            ),
            "openai/clip-vit-base-patch32 (processor)",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device)
        clip_model.eval()
        clip_alignment = {"model": clip_model, "processor": processor, "device": device}
        print(f"CLIP loaded on {device}!")
    return clip_alignment

def compute_image_text_similarity(prompt: str, image: Image.Image):
    """
    正确的 CLIP 语义相似度计算方法：
    1. 用 CLIP 编码提示词 -> text_features
    2. 用 CLIP 编码生成的图像 -> image_features
    3. 计算两者的余弦相似度 -> 这才是真正的语义匹配度
    """
    clip = get_clip()
    
    inputs = clip["processor"](text=[prompt], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip["device"]) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = clip["model"](**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    similarity = (text_features * image_features).sum(dim=-1).item()
    
    return {
        "similarity_score": float(similarity),
        "interpretation": "相似度越高表示生成的图像与提示词语义匹配度越好"
    }

def get_controlnet():
    global controlnet
    if controlnet is None:
        unload_all_sd15()
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        print("Loading ControlNet (Canny)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        controlnet = _hf_local_first(
            lambda lfo: ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                token=HF_TOKEN,
                local_files_only=lfo,
            ),
            "lllyasviel/sd-controlnet-canny",
        )

        print("Loading SD 1.5 base model...")
        cn = controlnet
        pipe = _hf_local_first(
            lambda lfo: StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=cn,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                token=HF_TOKEN,
                local_files_only=lfo,
            ),
            "runwayml/stable-diffusion-v1-5 (ControlNet pipeline)",
        )
        pipe = pipe.to(device)
        
        controlnet = {
            "pipeline": pipe,
            "device": device
        }
        print(f"ControlNet loaded on {device}!")
    return controlnet

def _unload_sd21_for_sd15() -> None:
    global image_generator
    if image_generator is not None:
        del image_generator
        image_generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_sd15_anime() -> None:
    global sd15_anime_pipe
    if sd15_anime_pipe is not None:
        del sd15_anime_pipe
        sd15_anime_pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_sd15_dreamshaper_full() -> None:
    global sd15_dreamshaper_pipe, sd15_dreamshaper_lora_style
    sd15_dreamshaper_lora_style = None
    if sd15_dreamshaper_pipe is not None:
        try:
            sd15_dreamshaper_pipe.unload_lora_weights()
        except Exception as e:
            print(f"unload_lora_weights: {e}")
        del sd15_dreamshaper_pipe
        sd15_dreamshaper_pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def unload_all_sd15() -> None:
    unload_sd15_anime()
    unload_sd15_dreamshaper_full()

def _resolve_local_path(rel: str | None) -> str | None:
    if not rel:
        return None
    p = os.path.join(_PROJECT_DIR, rel.replace("/", os.sep))
    return p if os.path.isfile(p) else None

def _get_lora_weight_file_path(meta: dict) -> str:
    """DreamShaper 风格 LoRA：优先项目内 local_fallback，否则 HF 缓存或按需下载。"""
    local = _resolve_local_path(meta.get("local_fallback"))
    if local:
        return local
    from huggingface_hub import hf_hub_download

    repo = meta["hf_repo"]
    fname = meta["weight_name"]

    def _dl(lfo: bool):
        return hf_hub_download(repo, fname, token=HF_TOKEN or None, local_files_only=lfo)

    try:
        return _dl(True)
    except Exception as e:
        if LOCAL_FILES_ONLY or not ALLOW_HF_HUB_DOWNLOAD:
            raise RuntimeError(
                f"LoRA 不在 HF 缓存 ({repo} / {fname})。请先下载或设置 ALLOW_HF_HUB_DOWNLOAD=true。"
            ) from e
        return _dl(False)

def _get_counterfeit_single_file_path() -> str:
    cf = STYLE_CONFIG["counterfeit"]
    local = _resolve_local_path(cf.get("local_fallback"))
    if local:
        return local
    from huggingface_hub import hf_hub_download

    def _dl(lfo: bool):
        return hf_hub_download(
            cf["hf_repo"],
            cf["weight_file"],
            token=HF_TOKEN or None,
            local_files_only=lfo,
        )

    try:
        return _dl(True)
    except Exception as e:
        if LOCAL_FILES_ONLY or not ALLOW_HF_HUB_DOWNLOAD:
            raise RuntimeError(
                f"Counterfeit weights not in HF cache ({cf['hf_repo']} / {cf['weight_file']}). "
                "Pre-download or set ALLOW_HF_HUB_DOWNLOAD=true."
            ) from e
        return _dl(False)

def get_sd15_anime_pipe():
    global sd15_anime_pipe
    unload_sd15_dreamshaper_full()
    _unload_sd21_for_sd15()
    if sd15_anime_pipe is not None:
        return sd15_anime_pipe
    from diffusers import StableDiffusionPipeline
    path = _get_counterfeit_single_file_path()
    print(f"Loading Counterfeit SD1.5 (anime)...")
    sd15_anime_pipe = _hf_local_first(
        lambda lfo: StableDiffusionPipeline.from_single_file(
            path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=lfo,
        ),
        "Counterfeit SD1.5 (single-file)",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd15_anime_pipe = sd15_anime_pipe.to(device)
    if hasattr(sd15_anime_pipe, "enable_vae_tiling"):
        sd15_anime_pipe.enable_vae_tiling()
    print("Counterfeit SD1.5 loaded.")
    return sd15_anime_pipe

def get_sd15_dreamshaper_base():
    global sd15_dreamshaper_pipe
    if sd15_dreamshaper_pipe is None:
        import logging
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        from diffusers import StableDiffusionPipeline
        repo = STYLE_CONFIG["dreamshaper"]["hf_repo"]
        print(f"Loading DreamShaper ({repo})...")
        sd15_dreamshaper_pipe = _hf_local_first(
            lambda lfo: StableDiffusionPipeline.from_pretrained(
                repo,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                token=HF_TOKEN,
                local_files_only=lfo,
            ),
            f"DreamShaper ({repo})",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sd15_dreamshaper_pipe = sd15_dreamshaper_pipe.to(device)
        if hasattr(sd15_dreamshaper_pipe, "enable_vae_tiling"):
            sd15_dreamshaper_pipe.enable_vae_tiling()
        print("DreamShaper loaded.")
    return sd15_dreamshaper_pipe

def _apply_lora_to_dreamshaper(style_key: str):
    global sd15_dreamshaper_lora_style
    meta = STYLE_CONFIG["loras"][style_key]
    pipe = get_sd15_dreamshaper_base()
    if sd15_dreamshaper_lora_style == style_key:
        return pipe, meta
    if sd15_dreamshaper_lora_style is not None:
        try:
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"unload_lora_weights: {e}")
        sd15_dreamshaper_lora_style = None
    adapter = meta["adapter_name"]
    scale = float(meta.get("scale", 0.8))
    print(f"Loading LoRA adapter={adapter} ({style_key})...")
    if meta.get("prepend_unet_prefix"):
        # 部分旧版 LoRA 存的是 down_blocks...processor.to_*_lora，无 unet. 前缀，PEFT 无法识别
        from safetensors.torch import load_file

        path = _get_lora_weight_file_path(meta)
        state_dict = load_file(path)
        state_dict = {
            (f"unet.{k}" if not k.startswith("unet.") else k): v for k, v in state_dict.items()
        }
        pipe.load_lora_weights(state_dict, adapter_name=adapter)
    else:
        local = _resolve_local_path(meta.get("local_fallback"))
        if local:
            pipe.load_lora_weights(local, adapter_name=adapter)
        else:
            try:
                pipe.load_lora_weights(
                    meta["hf_repo"],
                    weight_name=meta["weight_name"],
                    adapter_name=adapter,
                    token=HF_TOKEN,
                    local_files_only=True,
                )
            except Exception as e:
                if LOCAL_FILES_ONLY or not ALLOW_HF_HUB_DOWNLOAD:
                    raise RuntimeError(
                        f"LoRA not in HF cache ({meta['hf_repo']} / {meta['weight_name']}). "
                        "Pre-download or set ALLOW_HF_HUB_DOWNLOAD=true."
                    ) from e
                pipe.load_lora_weights(
                    meta["hf_repo"],
                    weight_name=meta["weight_name"],
                    adapter_name=adapter,
                    token=HF_TOKEN,
                    local_files_only=False,
                )
    if hasattr(pipe, "set_adapters"):
        try:
            pipe.set_adapters([adapter], adapter_weights=[scale])
        except ValueError as e:
            if "not in the list of present adapters" in str(e):
                raise RuntimeError(
                    f"LoRA 已加载但未能注册到管线（常与权重格式不兼容有关）。"
                    f"请检查 {meta.get('hf_repo')} / {meta.get('weight_name')} 是否为 SD1.5+DreamShaper 可用的 Kohya/Diffusers LoRA。"
                ) from e
            raise
    sd15_dreamshaper_lora_style = style_key
    return pipe, meta

def prepare_sd15_pipe(style: str):
    """style: anime | watercolor | oil | realistic | sketch"""
    global sd15_dreamshaper_lora_style
    allowed = {"anime", "watercolor", "oil", "realistic", "sketch"}
    if style not in allowed:
        raise ValueError(f"sd15_style must be one of {allowed}")
    if style == "anime":
        unload_sd15_dreamshaper_full()
        _unload_sd21_for_sd15()
        return get_sd15_anime_pipe(), None
    unload_sd15_anime()
    _unload_sd21_for_sd15()
    if style == "realistic":
        pipe = get_sd15_dreamshaper_base()
        if sd15_dreamshaper_lora_style is not None:
            try:
                pipe.unload_lora_weights()
            except Exception as e:
                print(f"unload_lora_weights: {e}")
            sd15_dreamshaper_lora_style = None
        return pipe, None
    return _apply_lora_to_dreamshaper(style)

@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multimodal API is running", "status": "ok"}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/status")
def status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "sd21": image_generator is not None,
            "sd15_anime": sd15_anime_pipe is not None,
            "sd15_dreamshaper": sd15_dreamshaper_pipe is not None,
            "blip": captioner is not None,
            "clip": clip_alignment is not None,
            "controlnet": controlnet is not None,
        }
    }

@app.post("/enhance")
def enhance_prompt_api(req: EnhanceRequest):
    try:
        original_prompt = req.prompt
        enhanced = enhance_prompt(original_prompt)
        return {
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-canny")
async def process_canny_image(file: UploadFile = File(...)):
    try:
        import cv2
        import numpy as np
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 转换为 numpy 数组
        img_array = np.array(image)
        
        # 转为灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Canny 边缘检测
        low_threshold = 100
        high_threshold = 200
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 转为3通道
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        
        # 转回 PIL Image
        canny_image = Image.fromarray(edges)
        
        # 返回 base64
        img_buffer = io.BytesIO()
        canny_image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return {
            "canny_image": f"data:image/png;base64,{img_str}",
            "original_size": {"width": image.width, "height": image.height}
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clip-evaluate")
def clip_evaluate_api(req: CLIPEvaluateRequest):
    try:
        translated = translate_to_english(req.prompt)
        
        if req.image:
            image_data = base64.b64decode(req.image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            result = compute_image_text_similarity(translated, image)
        else:
            result = {"error": "请提供图片以计算语义相似度"}
        
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_image(req: GenerateRequest):
    try:
        mode = (req.generation_mode or "sd21").lower()
        if mode not in ("sd21", "sd15"):
            raise HTTPException(status_code=400, detail="generation_mode 须为 sd21 或 sd15")
        if mode == "sd15":
            if not req.sd15_style:
                raise HTTPException(status_code=400, detail="使用 sd15 时必须提供 sd15_style")
            st = req.sd15_style.lower()
            if st not in ("anime", "watercolor", "oil", "realistic", "sketch"):
                raise HTTPException(status_code=400, detail="sd15_style 无效")

        if req.enhanced_prompt:
            translated_prompt = translate_to_english(req.enhanced_prompt)
            final_prompt = translated_prompt
        else:
            final_prompt = translate_to_english(req.prompt)
            translated_prompt = final_prompt

        num_steps = req.num_inference_steps
        guidance = req.guidance_scale
        
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        sd15_h = int(STYLE_CONFIG.get("sd15_size", 512))

        if mode == "sd15":
            pipe, lora_meta = prepare_sd15_pipe(st)
            if lora_meta:
                pfx = lora_meta.get("prompt_prefix_en")
                if pfx and pfx.lower() not in final_prompt.lower()[:80]:
                    final_prompt = f"{pfx} {final_prompt}"
                elif lora_meta.get("prompt_hint"):
                    hint = lora_meta["prompt_hint"]
                    if hint.lower() not in final_prompt.lower():
                        final_prompt = f"{final_prompt}, {hint}"
        else:
            pipe = get_generator()

        with torch.inference_mode():
            if mode == "sd21":
                image = pipe(
                    final_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    height=768,
                    width=768,
                    enable_vae_tiling=True
                ).images[0]
            else:
                image = pipe(
                    final_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                    height=sd15_h,
                    width=sd15_h,
                ).images[0]
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        clip_evaluation = compute_image_text_similarity(final_prompt, image)
        print(f"CLIP Image-Text Similarity: {clip_evaluation}")
        
        out = {
            "image": f"data:image/png;base64,{img_str}",
            "original_prompt": req.prompt,
            "translated_prompt": translated_prompt,
            "caption": caption,
            "clip_evaluation": clip_evaluation,
            "used_num_steps": num_steps,
            "used_guidance": guidance,
            "generation_mode": mode,
            "sd15_style": req.sd15_style if mode == "sd15" else None,
        }
        return out
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        return {"caption": caption}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-controlnet")
async def generate_with_controlnet(req: ControlNetGenerateRequest):
    try:
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        controlnet_data = get_controlnet()
        pipe = controlnet_data["pipeline"]
        
        image_data = base64.b64decode(req.control_image)
        control_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        control_image = control_image.resize((768, 768))
        
        if req.enhanced_prompt:
            translated = translate_to_english(req.enhanced_prompt)
        else:
            translated = translate_to_english(req.prompt)
        
        with torch.inference_mode():
            image = pipe(
                translated,
                image=control_image,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                controlnet_conditioning_scale=req.controlnet_conditioning_scale,
                height=768,
                width=768,
            ).images[0]
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=100)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        clip_evaluation = compute_image_text_similarity(translated, image)
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "original_prompt": req.prompt,
            "translated_prompt": translated,
            "caption": caption,
            "clip_evaluation": clip_evaluation,
            "used_num_steps": req.num_inference_steps,
            "used_guidance": req.guidance_scale
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
