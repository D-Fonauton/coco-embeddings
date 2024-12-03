import os
import hydra
from omegaconf import DictConfig
import pandas as pd


class PresentData:
    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)


    def get_identity(self, index):
        return self.data.name[index], self.data.subject[index], self.data.task[index]
    

    def get_position(self, index):
        return self.data.X[index], self.data.Y[index]
    

    def __len__(self):
        return len(self.data)


class COCOSearch18:
    def __init__(self, cfg_dataset) -> None:
        self.dir = cfg_dataset.dir
        self.conditions = cfg_dataset.conditions
        self.categories = cfg_dataset.categories
        
        self.images = self.load_images()

        # our big table data
        self.present = PresentData(cfg_dataset.present_path)


    # return the full path of selected image
    def full_path(self, condition, category, image_name):
        return os.path.join(self.dir, condition, category, image_name)


    # load all images in the category folder
    def load_category_images(self, condition, category):
        category_dir = os.path.join(self.dir, condition, category)
        return [os.path.join(category_dir, f) for f in os.listdir(category_dir) if f.endswith('.jpg')]
    

    # load all COCO-Search18 images
    def load_images(self):
        images = dict()
        for condition in self.conditions:
            images[condition] = dict()
            for category in self.categories:
                images[condition][category] = self.load_category_images(condition, category)
        return images
