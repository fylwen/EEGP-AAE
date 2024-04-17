# Edge Enhanced Implicit Orientation Learning with Geometric Prior for 6D Pose Estimation

Original implementation of the paper Yilin Wen, Hao Pan, Lei Yang and Wenping Wang, "Edge Enhanced Implicit Orientation Learning With Geometric Prior for 6D Pose Estimation", RAL, 2020, [[Paper]](https://ieeexplore.ieee.org/document/9126189).

This work improves the [Augmented AutoEncoder](https://github.com/DLR-RM/AugmentedAutoencoder) with a self-supervised scheme with the geometric prior imposed on the implicit orientation learning and a combination of color cue with the edge cue.

## Citiation
If you have interest with this work, please consider citing
```
@article{wen2020edge,
  title={Edge Enhanced Implicit Orientation Learning with Geometric Prior for 6D Pose Estimation},
  author={Wen, Yilin and Pan, Hao and Yang, Lei and Wang, Wenping},
  journal={IEEE Robotics and Automation Letters},
  year={2020},
  volume={5},
  number={3},
  pages={4931-4938},
  publisher={IEEE}
}
```

## Requirements

### The code is tested with the following environment
```  
Ubuntu 16.04
python 2.7
tensorflow 1.12.0
opencv-python 4.1.1
sonnet 1.23
open3d
numpy
matplotlib
scipy
pypng
ruamel.yaml
pillow
cython
pyopengl
glumpy
keras 2.2.5 (For Retina-Net)
```

### Pretrained Model

To run the demos of our pipeline, we provide pretrained ckpt files for T-LESS obj-29 and LINEMOD obj-14, and their corresponding pose embedding \bar_C for the inference stage in the following link [[Inference Data]](https://github.com/fylwen/EEGP-AAE/releases/download/assets-v1/assets.zip)

You may keep the downloaded folders under the root directory of this git repository:

1) ```./ws/experiments/```: The pretrained ckpt files.
2) ```./ws/meshes/```: 3D Meshes (Required for ICP).
3) ```./embedding92232s/```: Pose embedding \bar_C, 2D bounding boxes and rotations for sampled templates in the inference stage.
4) ```./demo_data/```: Data for demo images and 2D detections.

For T-LESS, we also provide its 2D detection backbone [RetinaNet](https://github.com/fizyr/keras-retinanet), with the pretrianed ckpt file provided by [pix2pose](https://github.com/kirumang/Pix2Pose) under the path ```./demo_data/resnet50_tless_19_inf.h5```


## Quick Start:

##### T-LESS Demo(RGB)

```python single3_test_RGB.py```

Estimate 6D pose with RetinaNet for a T-LESS box(obj-29) from an RGB image.

##### LINEMOD Demo(RGB-D)

```python single3_test_ICP.py```

Estimate 6D pose for a LINEMOD lamp(obj-14) from an RGB+D image and the Mask-RCNN result. 


## Train a Pose Estimation Network

### Prepared Training data

The rendering is based on [SIXD Toolkit](https://github.com/thodan/sixd_toolkit)

##### Render images for training the Auto-Encoder

```python render_training.py <obj_id> <batch_id>```

To render pair of augmented encoder input and decoder gt with randomly sampled rotations, this is to train the AE.

The code provides options to generate rendered data with batch process.

A sample of generated data for T-LESS obj-29 is provided under the path ```./ws/experiments/tmp_datasets/prepared_training_data_29_subdiv.npz```

##### Render images for pose embeddings

```python render_codebook.py <obj_id> <batch_id>```

To render canonical images for embeddings (applicable to both training and inference codebooks) with sampled rotations.

The code provides options to generate rendered data with batch process.

### Train with Generated data

After preparing the training data and the config file(e.g., ```./ws/cfg/subdiv_29_softmax_edge.cfg```, run ```single1_train.py``` to train our pose estimation network for the mesh.

```python single1_train.py <experiment_name> <obj_id>```

e.g., ```python single1_train.py subdiv_29_softmax_edge 29```


## Test with Trained Weights

### Generate Embedding \bar_C for inference

With trained ckpt, generate the \bar_C for rendered canonical images(could be generated by running ```python render_codebook.py```) under the sampled poses with

``` python single2_embed.py <experiment_name> <obj_id> <num_iterations>```

e.g., ```python single2_embed.py subdiv_29_softmax_edge 29 30000```


### 2D Detection

We have already integrated RetinaNet for T-LESS CAD models in ```single3_test_RGB.py```, we will release the integrated pipeline with Mask-RCNN for LINEMOD soon.

### 6D Pose Estimation

A basic pipeline for RGB-based and RGB+ICP-based 6D pose estimation is provided in ```single3_test_RGB.py``` and ```single3_test_ICP.py``` respectively, you may modify these two files to entertain your need.
