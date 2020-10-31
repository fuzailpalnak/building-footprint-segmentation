from building_footprint_segmentation.helpers import normalizer

from building_footprint_segmentation.ml.base_loader import BaseLoader
from building_footprint_segmentation.utils.operations import load_image
from building_footprint_segmentation.utils.py_network import to_input_image_tensor


class BinaryLoader(BaseLoader):
    def __init__(
        self, root_folder, image_normalization, ground_truth_normalization, mode
    ):
        super().__init__(
            root_folder, image_normalization, ground_truth_normalization, mode
        )

        self.image_normalization = getattr(normalizer, self.image_normalization)
        self.ground_truth_normalization = getattr(
            normalizer, self.ground_truth_normalization
        )

        # TODO transformation init

    def __len__(self):
        if len(self.images) != 0:
            return len(self.images)
        else:
            return len(self.labels)

    def __getitem__(self, idx):

        if self.mode in ["train", "val"]:
            image = load_image(str(self.images[idx]))
            ground_truth = load_image(str(self.labels[idx]))

            image = self.image_normalization(image)
            ground_truth = self.ground_truth_normalization(ground_truth)

            return {
                "images": to_input_image_tensor(image),
                "ground_truth": to_input_image_tensor(ground_truth),
            }

        elif self.mode == "test":
            image = load_image(str(self.images[idx]))
            return {
                "images": to_input_image_tensor(image),
                "file_name": str(self.images[idx]),
            }
        else:
            raise NotImplementedError
