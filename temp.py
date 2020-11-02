import io
import os
import re

from torch.utils import model_zoo

MODEL_URL = "https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip"
MODEL = ""

# state_dict = model_zoo.load_url(MODEL_URL, progress=True, map_location="cpu")
# print(state_dict)
current_dir = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        required = f.read().split("\n")
except FileNotFoundError:
    required = []

print(required)
