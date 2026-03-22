import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HF_HUB_DISABLE_XFF'] = '1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

HF_TOKEN = 'your-huggingface-token'

class ImageCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading BLIP captioning model (from cache): {model_id}")
        self.processor = BlipProcessor.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Image captioner loaded on {self.device}")
    
    def generate_caption(self, image, max_length=50, num_beams=5):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )  
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def generate_caption_with_prompt(self, image, prompt, max_length=50, num_beams=5):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams
            )
        
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def __call__(self, image, **kwargs):
        return self.generate_caption(image, **kwargs)


if __name__ == "__main__":
    captioner = ImageCaptioner()
    test_image = Image.new("RGB", (512, 512), color=(255, 128, 0))
    caption = captioner.generate_caption(test_image)
    print(f"Generated caption: {caption}")
