import numpy as np
from typing import List
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import multipers as mp
from multipers.filtrations import RipsCodensity
from multipers.filtrations.density import KDE


class KleinData:
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

    def sample_hybrid_klein_bottle(
        self, n=500, scale=1.0, hybrid_strength=0.3, noise_amp=0.1
    ):
        u = np.random.uniform(0, 2 * np.pi, n)
        v = np.random.uniform(0, 2 * np.pi, n)
        # Hybridized structure with periodic deformations
        x = np.cos(u) * (1 + 0.5 * np.sin(v)) + hybrid_strength * np.sin(2 * u + v)
        y = np.sin(u) * (1 + 0.5 * np.cos(v)) + hybrid_strength * np.cos(2 * u - v)
        z = 0.5 * np.cos(v) * np.sin(3 * u) + hybrid_strength * np.sin(u + v)
        w = 0.5 * np.sin(v) * np.cos(5 * u) + hybrid_strength * np.cos(2 * u + 2 * v)
        # Add some random surface roughness
        if noise_amp > 0:
            noise = np.random.normal(0, noise_amp, size=(n, 4))
        else:
            noise = 0
        return np.column_stack([x, y, z, w]) * scale + noise

    def sample_twist_klein_bottle(
        self, n=500, scale=1.0, twist=0.0, warp_freq=0.0, warp_amp=0.0
    ):
        u = np.random.uniform(0, 2 * np.pi, n)
        v = np.random.uniform(0, 2 * np.pi, n)
        # Apply twist to v depending on u
        v_twisted = v + twist * np.sin(u)
        # Base Klein bottle shape
        x = np.cos(u) * (1 + 0.5 * np.sin(v_twisted))
        y = np.sin(u) * (1 + 0.5 * np.sin(v_twisted))
        z = 0.5 * np.cos(v_twisted) * np.cos(u / 2)
        w = 0.5 * np.cos(v_twisted) * np.sin(u / 2)
        # Apply periodic warp
        if warp_amp > 0 and warp_freq > 0:
            x += warp_amp * np.sin(warp_freq * u)
            y += warp_amp * np.cos(warp_freq * v)
        # Scale
        points = np.column_stack([x, y, z, w]) * scale
        return points

    def sample_klein_bottle(self, n=500, scaled=0.8):
        u = np.random.uniform(0, 2 * np.pi, n)
        v = np.random.uniform(0, 2 * np.pi, n)
        x = np.cos(u) * (1 + 0.5 * np.sin(v))
        y = np.sin(u) * (1 + 0.5 * np.sin(v))
        z = 0.5 * np.cos(v) * np.cos(u / 2)
        w = 0.5 * np.cos(v) * np.sin(u / 2)
        return np.column_stack([scaled * x, scaled * y, scaled * z, scaled * w])

    def generate_dataset(self) -> Dataset:
        bottles = []
        modules = []
        pers_landscape = []
        for _ in range(self.num_data):
            # get torus
            scl = np.random.uniform(low=1, high=2)
            streg = np.random.uniform(low=1, high=2)
            amp = np.random.uniform(low=1, high=2)
            sch = np.random.uniform(low=0.5, high=2)
            tw = np.random.uniform(low=0.5, high=2)
            wf = np.random.uniform(low=0.5, high=2)
            wa = np.random.uniform(low=0.5, high=2)
            schal = np.random.uniform(low=0.5, high=2)

            m1 = self.sample_hybrid_klein_bottle(
                n=300, scale=scl, hybrid_strength=streg, noise_amp=amp
            )
            m2 = self.sample_twist_klein_bottle(
                n=300, scale=sch, twist=tw, warp_freq=wf, warp_amp=wa
            )
            m3 = self.sample_klein_bottle(n=200, scaled=schal)
            manifolds = [m1, m2, m3]
            for X in manifolds:
                X = X / np.max(np.abs(X))
                kb = torch.from_numpy(X).float()
                bottles.append(kb)
                # compute modules
                mods, lands = self.mods_and_landscapes(X)
                modules.append(mods)
                pers_landscape.append(lands)

        return KleinDataset(klein=bottles, modules=modules, landscapes=pers_landscape)

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


class KleinDataset(Dataset):
    def __init__(self, klein, modules, landscapes):
        self.klein = klein
        self.modules = modules
        self.landscapes = landscapes

    def __len__(self):
        return len(self.klein)

    def __getitem__(self, idx):
        # Return a tuple for the three columns
        return self.klein[idx], self.modules[idx], self.landscapes[idx]


def main():
    kleyn = KleinData(5, 200)
    dataset = kleyn.generate_dataset()
    torch.save(dataset, "klein_data.pt")


if __name__ == "__main__":
    main()
