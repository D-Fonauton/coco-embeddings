import clip
import torch

# Global cache for the model and preprocess function
_clip_cache = {}


def get_cached_clip(model_name: str, device: str):
    """Load and cache the model in RAM, and move it to the specified device if needed."""
    if model_name not in _clip_cache:
        print(f"Loading CLIP {model_name} on {device}... ", end="", flush=True)
        model, preprocess = clip.load(model_name, device=device)
        _clip_cache[model_name] = {
            "model": model,
            "preprocess": preprocess,
            "device": device,
        }
        print("done.")
    else:
        print(f"Pulling CLIP {model_name} from cache")

    # Move model to the correct device if not already there
    if _clip_cache[model_name]["device"] != device:
        print(f"Moving {model_name} to {device}... ", end="", flush=True)
        _clip_cache[model_name]["model"] = _clip_cache[model_name]["model"].to(device)
        _clip_cache[model_name]["device"] = device
        print("done.")

    return _clip_cache[model_name]["model"], _clip_cache[model_name]["preprocess"]


class CLIP:
    def __init__(self, cfg) -> None:
        self.device = cfg.clip.device
        self.model_name = cfg.clip.model_name
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)


    def calculate_similarity(self, text, image):
        text_input = clip.tokenize(text).to(self.device)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            return (image_features @ text_features.T).item()
