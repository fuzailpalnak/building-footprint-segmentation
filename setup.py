import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

current_dir = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
        install_requires = f.read().split("\n")
except FileNotFoundError:
    install_requires = []


setup(
    name="building-footprint-segmentation",
    version="0.2.4",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/fuzailpalnak/building-footprint-segmentation",
    description="Building footprint segmentation from satellite and aerial imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires="~=3.3",
    install_requires=install_requires,
    keywords=[
        "Deep Learning",
        "CNN",
        "Semantic Segmentation",
        "Building Footprint Extraction",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
)
