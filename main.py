import hydra
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
import numpy as np

# local
from src import CLIP, COCOSearch18
from utils import crop_image, crop_image_using_bbox


def gen_degree(number, cfg_dataset):
    return int(round(2 * np.tan(number / 2 / 180 * np.pi) * cfg_dataset.distance * cfg_dataset.width_pixel / cfg_dataset.width))


def calculate_simularity(cs18: COCOSearch18, cfg: DictConfig, model: CLIP, degree):
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
        crop_size = gen_degree(degree, cfg.dataset)
        image = crop_image(image, position, (crop_size, crop_size))
        text = cfg.clip.template.format(cfg.categories[cfg.dataset.categories.index(current_task)])
        
        simularity[index] = model.calculate_similarity(text, image)
    return simularity




def calculate_simularity_bbox(cs18: COCOSearch18, cfg: DictConfig, model: CLIP, mode):
    # using dataset config from config

    # image_names = np.unique([cs18.present.get_image_name(i) for i in range(len(cs18.present))])
    simularity = dict()
    for index in tqdm(range(len(cs18.present)), desc='calculating simularities'):
        current_name, current_subject, current_task = cs18.present.get_identity(index)
        bbox = cs18.present.get_bbox(index)

        image = Image.open(cs18.full_path("TP", current_task, current_name))
        image = crop_image_using_bbox(image, bbox, mode)

        text = cfg.clip.template.format(cfg.categories[cfg.dataset.categories.index(current_task)]) 
        simularity[index] = model.calculate_similarity(text, image)
    return simularity





@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    dataset = COCOSearch18(cfg.dataset)
    print(f'the length of present data: {len(dataset.present)}')
    present = dataset.present.data
    model = CLIP(cfg)

    """
    degrees = [5,10,15]
    for i in range(3):
        present[f"simularity_{degrees[i]}"] = calculate_simularity(dataset, cfg, model, degrees[i])
    """

    present[f"target_simularity_padding"] = calculate_simularity_bbox(dataset, cfg, model, "padding")
    present[f"target_simularity_fill"] = calculate_simularity_bbox(dataset, cfg, model, "fill")

    

    present.to_csv('new_present.csv')


if __name__ == '__main__':
    main()
