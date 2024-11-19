import hydra
from omegaconf import DictConfig
import clip
import torch
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Local
from src.models import get_cached_clip


def dimreduce(cfg, embeddings):
    reducer = UMAP(
        n_neighbors=cfg.umap.n_neighbors,
        n_components=cfg.umap.n_components,
        metric=cfg.umap.metric,
        random_state=cfg.umap.random_state,
    )
    return reducer.fit_transform(embeddings)


def clip_embed(cfg, text):
    device = cfg.clip.device
    model, preprocess = get_cached_clip(cfg.clip.model_name, device)
    tokenized_text = clip.tokenize(text).to(device)

    # Embed each category
    embeddings = []
    model.eval()
    with torch.no_grad():
        for tokens in tqdm(tokenized_text, desc="Embedding categories"):
            tokens = tokens.unsqueeze(0)  # Add batch dimension. could batch as well..
            text_features = model.encode_text(tokens).cpu().numpy()
            embeddings.append(text_features)

    return np.vstack(embeddings)  # Shape: (num_categories, embedding_dim)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Hydra config:", cfg)

    cats = cfg.categories
    inputs = [cfg.clip.template.format(cat) for cat in cats]
    embeddings = clip_embed(cfg, inputs)
    assert embeddings.shape[0] == len(cats)

    embeddings_2d = dimreduce(cfg, embeddings)
    plot_embeddings(embeddings_2d, cats)


def plot_embeddings(embeddings_2d, labels):
    plt.figure(figsize=(10, 6))

    # Plot each point and add annotation
    for i, label in enumerate(labels):
        x, y = embeddings_2d[i]
        plt.scatter(x, y, s=200)
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(10, 10),  # Offset for better visibility
            ha="center",
            fontsize=10,
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.6
            ),
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
