import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import numpy as np

# local
from src import CLIP, COCOSearch18
from utils import crop_image


def calculate_simularity(cs18: COCOSearch18, cfg: DictConfig, model: CLIP, crop_size):
    # using dataset config from config
    simularity = dict()
    for index in tqdm(range(len(cs18.present)), desc='calculating simularities'):
        current_name, current_subject, current_task = cs18.present.get_identity(index)
        image = Image.open(cs18.full_path("TP", current_task, current_name))
        x, y = cs18.present.get_position(index)
        
        # skip if x is nan
        if np.isnan(x):
            simularity[index] = np.nan
            continue

        # skip if fixation point is out of image border
        if x < 0 or x >= cfg.dataset.width_pixel or y < 0 or y >= cfg.dataset.height_pixel:
            simularity[index] = np.nan
            continue

        position = (round(x), round(y))
        image = crop_image(image, position, crop_size)
        text = cfg.clip.template.format(cfg.categories[cfg.dataset.categories.index(current_task)])
        
        simularity[index] = model.calculate_similarity(text, image)
    return simularity


def gen_degree(number, cfg_dataset):
    return int(round(2 * np.tan(number / 2 / 180 * np.pi) * cfg_dataset.distance * cfg_dataset.width_pixel / cfg_dataset.width))


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    degrees = [5,10,15]
    dataset = COCOSearch18(cfg.dataset)
    print(f'the length of present data: {len(dataset.present)}')
    present = dataset.present.data
    model = CLIP(cfg)

    for i in range(3):
        print(f"degrees: {degrees[i]}")
        crop_size = gen_degree(degrees[i], cfg.dataset)
        simularities = calculate_simularity(dataset, cfg, model, crop_size)
        present[f"simularity_{degrees[i]}"] = simularities

    present.to_csv('new_present.csv')


if __name__ == '__main__':
    main()
