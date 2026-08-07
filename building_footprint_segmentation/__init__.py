from building_footprint_segmentation._env import configure_windows_openmp

configure_windows_openmp()

import logging

logger = logging.getLogger("segmentation")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
