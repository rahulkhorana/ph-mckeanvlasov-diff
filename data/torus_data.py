import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import multipers as mp

from multipers.filtrations import RipsCodensity
from multipers.filtrations.density import KDE

import gudhi


class TorusData:
    def __init__(self, num_tori: int, num_points: int = 500):
        self.num_data = num_tori
        self.n_pt = num_points

    def sample_3_torus(self, n=500, r1=1.0, r2=0.2, r3=1.5, noise_std=0.0):
        t1 = np.random.uniform(0, 2 * np.pi, n)
        t2 = np.random.uniform(0, 2 * np.pi, n)
        t3 = np.random.uniform(0, 2 * np.pi, n)
        # Coordinates for the 3-torus
        x1 = (r1 + r2 * np.cos(t2)) * np.cos(t1)
        y1 = (r1 + r2 * np.cos(t2)) * np.sin(t1)
        x2 = (r3 + r2 * np.sin(t2)) * np.cos(t3)
        y2 = (r3 + r2 * np.sin(t2)) * np.sin(t3)
        x3 = r2 * np.cos(t2) * np.sin(t3)
        y3 = r2 * np.sin(t2) * np.cos(t3)
        X = np.column_stack([x1, y1, x2, y2, x3, y3])
        if noise_std > 0:
            X += np.random.normal(0, noise_std, size=X.shape)
        return X

    def extract_np_arrays(self, data):
        """
        Recursively extracts all NumPy arrays from a nested data structure.
        """
        if isinstance(data, np.ndarray):
            yield data
        elif isinstance(data, (list, tuple)):
            for item in data:
                yield from self.extract_np_arrays(item)

    def generate_dataset(self) -> Dataset:
        tori = []
        modules = []
        pers_landscape = []
        for _ in range(self.num_data):
            # get torus
            r1 = np.random.uniform(low=0.1, high=2)
            r2 = np.random.uniform(low=0.1, high=2)
            r3 = np.random.uniform(low=0.1, high=2)
            noise = np.random.uniform(low=0.1, high=1)
            X = self.sample_3_torus(self.n_pt, r1, r2, r3, noise)
            X = X / np.max(np.abs(X))
            torus = torch.from_numpy(X).float()
            tori.append(torus)
            # compute modules
            avg_radius = np.mean([r1, r2, r3])
            bandwidth = float(avg_radius / 10.0)
            density = KDE(bandwidth=bandwidth).fit(X).score_samples(X)
            lower, upper = np.percentile(density, [5, 95])
            codensity = 1 - np.clip((density - lower) / (upper - lower), 0, 1)
            simplextree = RipsCodensity(
                points=X,
                bandwidth=bandwidth,
                threshold_radius=0.6,
                kernel="gaussian",
                return_log=False,
            )
            simplextree.expansion(3)
            bimod = mp.module_approximation(simplextree)
            module_data = bimod.dump()
            module_tensors = [
                torch.from_numpy(arr).float()
                for arr in self.extract_np_arrays(module_data)
            ]
            modules.append(module_tensors)
            # get landscapes
            landscape_0 = bimod.landscapes(
                degree=0,
                ks=range(3),
                plot=False,
                box=bimod.get_box(),
                resolution=(100, 100),
            )

            landscape_1 = bimod.landscapes(
                degree=1,
                ks=range(3),
                plot=False,
                box=bimod.get_box(),
                resolution=(100, 100),
            )

            landscape_2 = bimod.landscapes(
                degree=2,
                ks=range(3),
                plot=False,
                box=bimod.get_box(),
                resolution=(100, 100),
            )
            landscape_0 = torch.from_numpy(landscape_0).float()
            landscape_1 = torch.from_numpy(landscape_1).float()
            landscape_2 = torch.from_numpy(landscape_2).float()
            lands = torch.stack([landscape_0, landscape_1, landscape_2], dim=0).float()
            pers_landscape.append(lands)

        return TorusDataset(tori=tori, modules=modules, landscapes=pers_landscape)


class TorusDataset(Dataset):
    def __init__(self, tori, modules, landscapes):
        self.tori = tori
        self.modules = modules
        self.landscapes = landscapes

    def __len__(self):
        return len(self.tori)

    def __getitem__(self, idx):
        # Return a tuple for the three columns
        return self.tori[idx], self.modules[idx], self.landscapes[idx]


def main():
    tor = TorusData(10, 300)
    dataset = tor.generate_dataset()
    torch.save(dataset, "torus_data.pt")


if __name__ == "__main__":
    main()
