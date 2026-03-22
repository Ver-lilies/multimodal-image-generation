import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HF_HUB_DISABLE_XFF'] = '1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

HF_TOKEN = 'your-huggingface-token'

class ImageGenerator:
    def __init__(self, model_id="sd2-community/stable-diffusion-2-1-base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"\n{'='*60}")
        print(f"🔄 正在加载 Stable Diffusion 模型 (本地缓存)")
        print(f"{'='*60}\n")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            token=HF_TOKEN,
            local_files_only=True
        )
        
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print(f"\n✅ Image generator loaded on {self.device}")
    
    def generate(self, prompt, num_inference_steps=30, guidance_scale=7.5, height=512, width=512):
        with torch.inference_mode():
            image = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            ).images[0]
        return image
    
    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


if __name__ == "__main__":
    generator = ImageGenerator()
    test_image = generator.generate("A beautiful sunset over the ocean")
    test_image.save("test_output.png")
    print("Test image saved to test_output.png")
