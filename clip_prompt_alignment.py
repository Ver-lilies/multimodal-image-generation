import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:2334'
os.environ['HF_HUB_DISABLE_XFF'] = '1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

import torch
from transformers import CLIPProcessor, CLIPModel

HF_TOKEN = 'your-huggingface-token'

class CLIPPromptAlignment:
    def __init__(self, model_id="openai/clip-vit-base-patch32", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"\n{'='*60}")
        print(f"🔄 正在加载 CLIP 模型")
        print(f"{'='*60}\n")

        self.model = CLIPModel.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
        self.processor = CLIPProcessor.from_pretrained(model_id, token=HF_TOKEN, local_files_only=True)
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"\n✅ CLIP model loaded on {self.device}")

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def compute_similarity(self, text1, text2):
        features1 = self.encode_text(text1)
        features2 = self.encode_text(text2)

        similarity = (features1 * features2).sum(dim=-1)
        return similarity.item()

    def evaluate_prompt_quality(self, prompt, target_concept=None):
        scores = {}

        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scores['text_length'] = len(prompt.split())
        scores['has_details'] = any(word in prompt.lower() for word in ['color', 'style', 'light', 'background', 'detailed'])

        if target_concept:
            target_features = self.encode_text(target_concept)
            similarity = (text_features * target_features).sum(dim=-1).item()
            scores['concept_similarity'] = similarity

        return scores

    def suggest_improvements(self, prompt, target_concept=None):
        suggestions = []

        if len(prompt.split()) < 5:
            suggestions.append("提示词太短，建议添加更多细节描述")

        common_missing_words = ['color', 'style', 'lighting', 'background', 'detailed', 'beautiful', 'high quality']
        missing = [word for word in common_missing_words if word not in prompt.lower()]
        if missing:
            suggestions.append(f"可考虑添加: {', '.join(missing[:3])}")

        if target_concept:
            scores = self.evaluate_prompt_quality(prompt, target_concept)
            if scores.get('concept_similarity', 0) < 0.2:
                suggestions.append("提示词与目标概念相关性较低")

        return suggestions if suggestions else ["提示词质量良好"]


if __name__ == "__main__":
    clip = CLIPPromptAlignment()

    test_prompt = "A cute cat sitting on a table"
    print(f"\n测试提示词: {test_prompt}")

    scores = clip.evaluate_prompt_quality(test_prompt)
    print(f"质量评估: {scores}")

    suggestions = clip.suggest_improvements(test_prompt)
    print(f"改进建议: {suggestions}")
