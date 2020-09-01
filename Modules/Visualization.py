import matplotlib.pyplot as plt

import numpy as np

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable

import os

# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/media/juancr/Data/2020Lyft/lyft-motion-prediction-autonomous-vehicles"


class VisualizationClass():
    dataset_path = []
    zarr_dataset = []
    rast = 0
    dataset = []
    cfg = 0

    def __init__(self):
        print("Visualization Class initialized.")
        # get config
        self.cfg = load_config_data("/mnt/extra/kaggle/competitions/2020lyft/ProjectLyft/Modules/visualisation_config.yaml")
        print(self.cfg)

        dm = LocalDataManager()
        self.dataset_path = dm.require(self.cfg["val_data_loader"]["key"])
        self.zarr_dataset = ChunkedDataset(self.dataset_path)
        self.zarr_dataset.open()


        # Dataset package
        self.rast = build_rasterizer(self.cfg, dm)
        self.dataset = EgoDataset(self.cfg, self.zarr_dataset, self.rast)

    def loadData(self):
        print(self.zarr_dataset)

    def workingRawData(self):
        frames = self.zarr_dataset.frames
        coords = np.zeros((len(frames), 2))
        for idx_coord, idx_data in enumerate(tqdm(range(len(frames)), desc="getting centroid to plot trajectory")):
            frame = self.zarr_dataset.frames[idx_data]
            coords[idx_coord] = frame["ego_translation"][:2]

        plt.scatter(coords[:, 0], coords[:, 1], marker='.')
        axes = plt.gca()
        axes.set_xlim([-2500, 1600])
        axes.set_ylim([-2500, 1600])
        plt.show()

    def agentsTable(self):
        agents = self.zarr_dataset.agents
        probabilities = agents["label_probabilities"]
        labels_indexes = np.argmax(probabilities, axis=1)
        counts = []
        for idx_label, label in enumerate(PERCEPTION_LABELS):
            counts.append(np.sum(labels_indexes == idx_label))

        table = PrettyTable(field_names=["label", "counts"])
        for count, label in zip(counts, PERCEPTION_LABELS):
            table.add_row([label, count])
        print(table)

    def visualizeAV(self):
        data = self.dataset[50]

        im = data["image"].transpose(1, 2, 0)
        im = self.dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2],
                                                   data["world_to_image"])
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

        plt.imshow(im[::-1])
        plt.show()

    def visualizeAgent(self):
        self.dataset = AgentDataset(self.cfg, self.zarr_dataset, self.rast)
        data = self.dataset[0]

        im = data["image"].transpose(1, 2, 0)
        im = self.dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2],
                                                   data["world_to_image"])
        draw_trajectory(im, target_positions_pixels, data["target_yaws"], TARGET_POINTS_COLOR)

        plt.imshow(im[::-1])
        plt.show()

