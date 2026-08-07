from typing import Any

from pathlib import Path

import torch
from albumentations import Compose
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class Loader:
    train_loader: Any
    val_loader: Any
    test_loader: Any


class BaseLoader(Dataset):
    def __init__(
        self,
        root_folder: str,
        image_normalization: str,
        ground_truth_normalization: str,
        augmenters: Compose,
        mode: str,
    ):
        self.mode = mode
        self.root_folder = Path(root_folder)

        self.image_normalization = image_normalization
        self.ground_truth_normalization = ground_truth_normalization
        self.augmenters = augmenters

        self.images = self._list_image_files(self.root_folder / self.mode / "images")
        self.labels = self._list_image_files(self.root_folder / self.mode / "labels")

    @staticmethod
    def _list_image_files(folder: Path) -> list[Path]:
        if not folder.exists():
            return []
        return sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    @classmethod
    def _make_split_loader(
        cls,
        root_folder: str,
        image_normalization: str,
        label_normalization: str,
        augmenters: Compose,
        mode: str,
        batch_size: int,
    ) -> DataLoader:
        dataset = cls(
            root_folder,
            image_normalization,
            label_normalization,
            augmenters,
            mode,
        )
        return DataLoader(
            dataset=dataset,
            shuffle=len(dataset) > 0,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def get_data_loader(
        cls,
        root_folder: str,
        image_normalization: str,
        label_normalization: str,
        augmenters: Compose,
        batch_size: int,
    ):
        train_data = cls._make_split_loader(
            root_folder,
            image_normalization,
            label_normalization,
            augmenters,
            "train",
            batch_size,
        )
        val_data = cls._make_split_loader(
            root_folder,
            image_normalization,
            label_normalization,
            Compose([]),
            "val",
            batch_size,
        )
        test_data = cls._make_split_loader(
            root_folder,
            image_normalization,
            label_normalization,
            Compose([]),
            "test",
            batch_size,
        )
        return Loader(train_data, val_data, test_data)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
