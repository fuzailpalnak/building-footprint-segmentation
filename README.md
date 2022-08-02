# Building Footprint Segmentation

#### Library to train building footprint on satellite and aerial imagery.

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Licence](https://img.shields.io/github/license/fuzailpalnak/building-footprint-segmentation)
![Downloads](https://pepy.tech/badge/building-footprint-segmentation)


<a href='https://ko-fi.com/fuzailpalnak' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

![merge1](https://user-images.githubusercontent.com/24665570/97859410-91fa6100-1d26-11eb-8a47-e41982c748d7.jpg)



## Installation
    
    
    pip install building-footprint-segmentation
    

## Dataset 

- [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/)
- [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

## Training

- [Train With Config](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20config.ipynb)
    , Use [config template](https://codebeautify.org/yaml-validator/cbc60637) for generating training config

- [Train With Arguments](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20defined%20arguments.ipynb)

## Visualize Training

##### Test images at end of every epoch

- Follow [Example](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/TestCallBack.ipynb)
 
##### Visualizing on Tensorboard

```python
from building_footprint_segmentation.helpers.callbacks import CallbackList, TensorBoardCallback
where_to_log_the_callback = r"path_to_log_callback"   
callbacks = CallbackList()

# Ouptut from all the callbacks caller will be stored at the path specified in log_dir
callbacks.append(TensorBoardCallback(where_to_log_the_callback))

```

To view Tensorboard dash board

    tensorboard --logdir="path_to_log_callback"

## Defining Custom Callback
```python
from building_footprint_segmentation.helpers.callbacks import CallbackList, Callback

class CustomCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)


where_to_log_the_callback = r"path_to_log_callback"   
callbacks = CallbackList()

# Ouptut from all the callbacks caller will be stored at the path specified in log_dir
callbacks.append(CustomCallback(where_to_log_the_callback))
```

## Split the images in smaller sample
```python
import glob
import os

from image_fragment.fragment import ImageFragment

# FOR .jpg, .png, .jpeg
from imageio import imread, imsave

# FOR .tiff
from tifffile import imread, imsave

ORIGINAL_DIM_OF_IMAGE = (1500, 1500, 3)
CROP_TO_DIM = (384, 384, 3)

image_fragment = ImageFragment.image_fragment_3d(
    fragment_size=(384, 384, 3), org_size=ORIGINAL_DIM_OF_IMAGE
)

IMAGE_DIR = r"pth\to\input\dir"
SAVE_DIR = r"pth\to\save\dir"

for file in glob.glob(
    os.path.join(IMAGE_DIR, "*")
):
    image = imread(file)
    for i, fragment in enumerate(image_fragment):
        # GET DATA THAT BELONGS TO THE FRAGMENT
        fragmented_image = fragment.get_fragment_data(image)

        imsave(
            os.path.join(
                SAVE_DIR,
                f"{i}_{os.path.basename(file)}",
            ),
            fragmented_image,
        )

```
## Segmentation for building footprint

- [x] binary
- [ ] building with boundary (multi class segmentation)

## Weight File

- [RefineNet trained on INRIA](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip)
- [DlinkNet trained on Massachusetts Buildings Dataset](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/DlinkNet.zip)

## Commonly used utility task when working with Geotiff

- [Generate bitmap from shape file](https://github.com/fuzailpalnak/py-gis-utility#generate-bitmap-from-shape-file)
- [Generate shape geometry from geo reference bitmap](https://github.com/fuzailpalnak/py-gis-utility#generate-shape-geometry-from-geo-reference-bitmap)
- [Save Multi Band Imagery](https://github.com/fuzailpalnak/py-gis-utility#save-multi-band-imagery)