import clip

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
