# Spleen_injury_detection

# Introduction
This repository contains the source code for classificating the spleen injury and visualizing by heatmaps from contrast-enhanced CT.

# Prerequisites

* Ubuntu: 18.04 lts
* Python 3.6.9
* Pytorch 1.6.0
* Keras 2.2.4
* NVIDIA GPU + CUDA_10.2 CuDNN_7.6.5
This repository has been tested on NVIDIA TITAN RTX.

# Installation

* pip install -r requirements.txt

# Usage

## Inference
### Spleen cropping
```
python3 crop_inference.py
```
### Spleen injury classification
```
python3 All_grad_cam_crop_torch_new_data.py

```
