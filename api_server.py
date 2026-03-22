import os
import torch
import base64
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

HF_TOKEN = 'your-huggingface-token'
DEEPSEEK_API_KEY = 'sk-fd4346c63f3d490799d425ea6a79ea15'

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

class GenerateRequest(BaseModel):
    prompt: str
    enhanced_prompt: str | None = None
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    clip_evaluation_score: float | None = None
    use_clip_optimization: bool = False

class EnhanceRequest(BaseModel):
    prompt: str

class CLIPEvaluateRequest(BaseModel):
    prompt: str

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

def get_translator():
    global translator
    if translator is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        print("Loading Translator (zh->en)...")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en", token=HF_TOKEN, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en", token=HF_TOKEN, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        translator = {"tokenizer": tokenizer, "model": model, "device": device}
        print(f"Translator loaded on {device}!")
    return translator

def translate_to_english(text: str) -> str:
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    if has_chinese:
        trans = get_translator()
        inputs = trans["tokenizer"](text, return_tensors="pt", padding=True).to(trans["device"])
        with torch.no_grad():
            outputs = trans["model"].generate(**inputs, max_length=512)
        translated = trans["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        print(f"Translated: '{text}' -> '{translated}'")
        return translated
    return text

def get_generator():
    global image_generator
    if image_generator is None:
        import logging
        logging.getLogger("diffusers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        from diffusers import StableDiffusionPipeline, DDIMScheduler
        import sys
        from io import StringIO
        
        print("Loading Stable Diffusion from cache...")
        image_generator = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            token=HF_TOKEN,
            local_files_only=True
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
        print("Loading BLIP from cache...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", token=HF_TOKEN, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", token=HF_TOKEN, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        captioner = {"processor": processor, "model": model}
        print(f"BLIP loaded on {device}!")
    return captioner

def get_clip():
    global clip_alignment
    if clip_alignment is None:
        from transformers import CLIPProcessor, CLIPModel
        print("Loading CLIP for prompt alignment...")
        clip_alignment = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=False)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", token=HF_TOKEN, local_files_only=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_alignment = clip_alignment.to(device)
        clip_alignment.eval()
        clip_alignment = {"model": clip_alignment, "processor": processor, "device": device}
        print(f"CLIP loaded on {device}!")
    return clip_alignment

def evaluate_prompt_with_clip(prompt, target_words=None):
    clip = get_clip()
    
    quality_refs = [
        "a detailed high quality realistic photograph",
        "a beautiful artistic illustration",
        "a clear detailed image with good composition",
        "a poor low quality blurry uncertain image"
    ]
    
    all_texts = [prompt] + quality_refs
    inputs = clip["processor"](text=all_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(clip["device"]) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_features = clip["model"].get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    prompt_feature = text_features[0:1]
    quality_features = text_features[1:]
    
    similarities = (prompt_feature * quality_features).sum(dim=-1).cpu().numpy()
    
    positive_score = (similarities[0] + similarities[1] + similarities[2]) / 3
    negative_score = similarities[3]
    
    quality_score = float((positive_score - negative_score + 1) / 2)
    quality_score = max(0.1, min(quality_score, 0.95))
    
    similarity_score = 0.5
    has_details = any(word in prompt.lower() for word in ['color', 'style', 'light', 'background', 'beautiful', 'detailed', 'high quality'])
    
    if target_words:
        target_inputs = clip["processor"](text=[target_words], return_tensors="pt", padding=True)
        target_inputs = {k: v.to(clip["device"]) for k, v in target_inputs.items()}
        with torch.no_grad():
            target_features = clip["model"].get_text_features(**target_inputs)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)
        similarity_score = (prompt_feature * target_features).sum(dim=-1).item()
    
    return {
        "quality_score": quality_score,
        "similarity_score": similarity_score,
        "word_count": len(prompt.split()),
        "has_details": has_details
    }

@app.get("/")
def root():
    return {"message": "Multimodal API is running", "status": "ok"}

@app.get("/status")
def status():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "sd": image_generator is not None,
            "blip": captioner is not None,
            "clip": clip_alignment is not None
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

@app.post("/clip-evaluate")
def clip_evaluate_api(req: CLIPEvaluateRequest):
    try:
        translated = translate_to_english(req.prompt)
        result = evaluate_prompt_with_clip(translated)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_image(req: GenerateRequest):
    try:

        if req.enhanced_prompt:
            translated_prompt = translate_to_english(req.enhanced_prompt)
            final_prompt = translated_prompt
            target_words = translate_to_english(req.prompt)
        else:
            final_prompt = translate_to_english(req.prompt)
            translated_prompt = final_prompt
            target_words = None

        clip_evaluation = evaluate_prompt_with_clip(final_prompt, target_words)
        print(f"CLIP Prompt Evaluation: {clip_evaluation}")

        num_steps = req.num_inference_steps
        guidance = req.guidance_scale
        
        if req.use_clip_optimization and req.clip_evaluation_score is not None:
            quality_score = req.clip_evaluation_score
            if quality_score < 0.5:
                num_steps = min(num_steps + 20, 100)
                guidance = min(guidance + 3, 20)
                print(f"CLIP Optimization: Low quality ({quality_score:.2f}), increasing steps to {num_steps}, guidance to {guidance}")
            elif quality_score < 0.7:
                num_steps = min(num_steps + 10, 100)
                guidance = min(guidance + 1.5, 20)
                print(f"CLIP Optimization: Medium quality ({quality_score:.2f}), increasing steps to {num_steps}, guidance to {guidance}")

        pipe = get_generator()
        
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        with torch.inference_mode():
            image = pipe(
                final_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                height=768,
                width=768,
                enable_vae_tiling=False
            ).images[0]
        
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        cap = get_captioner()
        inputs = cap["processor"](image, return_tensors="pt").to(cap["model"].device)
        with torch.no_grad():
            output = cap["model"].generate(**inputs, max_length=50)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        return {
            "image": f"data:image/png;base64,{img_str}",
            "original_prompt": req.prompt,
            "translated_prompt": translated_prompt,
            "caption": caption,
            "clip_evaluation": clip_evaluation,
            "used_num_steps": num_steps,
            "used_guidance": guidance
        }
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
            output = cap["model"].generate(**inputs, max_length=50)
        caption = cap["processor"].decode(output[0], skip_special_tokens=True)
        
        return {"caption": caption}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
