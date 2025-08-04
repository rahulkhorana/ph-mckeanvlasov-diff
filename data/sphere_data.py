import numpy as np
from typing import List
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import multipers as mp
from multipers.filtrations import RipsCodensity
from multipers.filtrations.density import KDE


class SphereData:
    def __init__(self, num_tori: int, num_points: int = 500):
        self.num_data = num_tori
        self.n_pt = num_points

    def extract_np_arrays(self, data):
        """
        Recursively extracts all NumPy arrays from a nested data structure.
        """
        if isinstance(data, np.ndarray):
            yield data
        elif isinstance(data, (list, tuple)):
            for item in data:
                yield from self.extract_np_arrays(item)

    def sample_param_sphere(self, n=1000, radius=1.0):
        theta = np.arccos(1 - 2 * np.random.rand(n))  # [0, pi]
        phi = 2 * np.pi * np.random.rand(n)  # [0, 2pi]
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return np.column_stack([x, y, z])

    def sample_bumpy_sphere(self, n=1000, base_radius=1.0, bump_freq=5.0, bump_amp=0.2):
        theta = np.arccos(1 - 2 * np.random.rand(n))
        phi = 2 * np.pi * np.random.rand(n)
        r = base_radius + bump_amp * np.sin(bump_freq * theta) * np.cos(bump_freq * phi)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.column_stack([x, y, z])

    def sample_sphere_pack(
        self, n_spheres=5, points_per_sphere=200, spread=3.0, bump=False
    ):
        points = []
        for _ in range(n_spheres):
            center = np.random.uniform(-spread, spread, size=3)
            if bump:
                sphere = self.sample_bumpy_sphere(n=points_per_sphere)
            else:
                sphere = self.sample_param_sphere(n=points_per_sphere)
            sphere += center
            points.append(sphere)
        return np.vstack(points)

    def generate_dataset(self) -> Dataset:
        spheres = []
        modules = []
        pers_landscape = []
        for _ in range(self.num_data):
            # get torus
            r1 = np.random.uniform(low=0.1, high=10)
            bump_freq = np.random.uniform(low=0.1, high=10)
            amp = np.random.uniform(low=0.1, high=1)
            spread = r1 = np.random.uniform(low=0.1, high=10)
            m1 = self.sample_param_sphere(n=200, radius=r1)
            m2 = self.sample_bumpy_sphere(n=200, bump_freq=bump_freq, bump_amp=amp)
            m3 = self.sample_sphere_pack(
                n_spheres=3, points_per_sphere=200, spread=spread, bump=False
            )
            m4 = self.sample_sphere_pack(
                n_spheres=3, points_per_sphere=200, spread=spread, bump=True
            )
            manifolds = [m1, m2, m3, m4]
            for X in manifolds:
                sphere = torch.from_numpy(X).float()
                spheres.append(sphere)
                # compute modules
                mods, lands = self.mods_and_landscapes(X)
                modules.append(mods)
                pers_landscape.append(lands)

        return SphereDataset(
            spheres=spheres, modules=modules, landscapes=pers_landscape
        )

    def mods_and_landscapes(self, X) -> List:
        density = KDE(bandwidth=0.2).fit(X).score_samples(X)
        lower, upper = np.percentile(density, [5, 95])
        simplextree = RipsCodensity(
            points=X,
            bandwidth=0.2,
            threshold_radius=0.6,
            kernel="gaussian",
            return_log=False,
        )
        simplextree.expansion(2)
        bimod = mp.module_approximation(simplextree)
        module_data = bimod.dump()
        module_tensors = [
            torch.from_numpy(arr).float() for arr in self.extract_np_arrays(module_data)
        ]
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
        return [module_tensors, lands]


class SphereDataset(Dataset):
    def __init__(self, spheres, modules, landscapes):
        self.spheres = spheres
        self.modules = modules
        self.landscapes = landscapes

    def __len__(self):
        return len(self.spheres)

    def __getitem__(self, idx):
        # Return a tuple for the three columns
        return self.spheres[idx], self.modules[idx], self.landscapes[idx]


def main():
    spherz = SphereData(5, 500)
    dataset = spherz.generate_dataset()
    torch.save(dataset, "sphere_data.pt")


if __name__ == "__main__":
    main()
