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

## Segmentation for building footprint

- [x] binary
- [ ] building with boundary (multi class segmentation)

## Weight File

- [RefineNet](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip)
