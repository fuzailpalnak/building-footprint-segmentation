# Building Footprint Segmentation

#### Library to train building footprint on satellite and aerial imagery.

![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Licence](https://img.shields.io/github/license/fuzailpalnak/building-footprint-segmentation)

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

##### Test images at end of every epoch. Follow [Example](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/TestCallBack.ipynb)

```python
from building_footprint_segmentation.helpers.callbacks import CallbackList, TestDuringTrainingCallback

class TestCallback(TestDuringTrainingCallback):
    def inference(self, model, image, file_name, save_path, index):
        """
        
        :param model: the model used for training
        :param image: the images loaded by the test loader
        :param file_name: the file name of the test image
        :param save_path: path where to save the image
        :param index: 
        :return: 
        """
        # Define this method on how to handle the prediction at the end of every epoch

where_to_log_the_callback = r"path_to_log_callback"   
callbacks = CallbackList()

# Ouptut from all the callbacks caller will be stored at the path specified in log_dir
callbacks.append(TestCallback(where_to_log_the_callback))
```

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



## Segmentation for building footprint

- [x] binary
- [ ] building with boundary (multi class segmentation)

## Weight File

- [RefineNet](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip)