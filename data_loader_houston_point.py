import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from scipy.io import loadmat


def normalize_channels(x):
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x[:, :, None]
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[2]):
        c = x[:, :, i]
        cmin, cmax = c.min(), c.max()
        if cmax > cmin:
            out[:, :, i] = (c - cmin) / (cmax - cmin)
        else:
            out[:, :, i] = 0
    return out


class HoustonPointDataset(TorchDataset):
    def __init__(self, root_dir, split='train', patch_size=32):
        """
        root_dir:
            /mnt/mydisk/ys/Houston2013
        split:
            'train' or 'test'
        """
        self.root_dir = root_dir
        self.split = split
        self.patch_size = patch_size
        self.pad = patch_size // 2

        # 读原始数据
        hsi = loadmat(os.path.join(root_dir, 'Houston.mat'))['Houston']          # (1905, 349, 144)
        lidar = loadmat(os.path.join(root_dir, 'LiDAR_1.mat'))['LiDAR'].T        # 转成 (1905, 349)
        train_gt = loadmat(os.path.join(root_dir, 'TRAIN_GT.mat'))['TR']         # (1905, 349)
        test_gt = loadmat(os.path.join(root_dir, 'TEST_GT.mat'))['TE']           # (1905, 349)

        self.hsi = normalize_channels(hsi)
        self.lidar = normalize_channels(lidar)

        self.gt = train_gt if split == 'train' else test_gt

        # 只保留官方标注点
        coords = np.argwhere(self.gt > 0)
        self.coords = coords

        # padding，确保边界点也能取到 32x32 patch
        self.hsi_pad = np.pad(
            self.hsi,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='reflect'
        )
        self.lidar_pad = np.pad(
            self.lidar,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='reflect'
        )

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]

        rp, cp = r + self.pad, c + self.pad

        hsi_patch = self.hsi_pad[
            rp - self.pad: rp + self.pad,
            cp - self.pad: cp + self.pad,
            :
        ]
        lidar_patch = self.lidar_pad[
            rp - self.pad: rp + self.pad,
            cp - self.pad: cp + self.pad,
            :
        ]

        # 标签从 1..15 转成 0..14
        label = int(self.gt[r, c]) - 1

        hsi_patch = torch.from_numpy(hsi_patch).permute(2, 0, 1).float()
        lidar_patch = torch.from_numpy(lidar_patch).permute(2, 0, 1).float()
        label = torch.tensor(label, dtype=torch.long)

        return hsi_patch, lidar_patch, label