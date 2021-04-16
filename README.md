# Human Pose Estimation on COCO Dataset

This repository contains the source code for training a deep neural network to perform human pose estimation from scratch. A brief excerpt from our final report is copied below.

## Getting Started

- init submodules
```
git submodule update --init --recursive
```
- install requirements
```
pip3 install -r requirements.txt
```
- install pycocotools manually if `requirements.txt` failed
```
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
- install dataset (requires just under 30 GB after unzipping, but will require at least 45 GB free disk space to unzip a ~18 GB training set)
```
bash ./scripts/coco_dl.sh
```

## Troubleshooting
pycocotools: https://stackoverflow.com/questions/49311195/how-to-install-coco-pythonapi-in-python3

## Abstract

Human pose estimation (HPE) is the task of identifying body keypoints on an input image to construct a body model. The motivation for this topic was driven by the exciting applications of HPE: pedestrian behaviour detection, sign language translation, animation and film, security systems, sports science, and many others. HPE shares many challenges with typical computer vision problems, such as intra-class variations, lighting, perspective, and object occlusions. It also faces challenges unique to HPE such as strong articulations, small and barely visible joints, and self-occlusions from overlapping joints. This report discusses a stacked hourglass network architecture that was developed and trained from scratch to achieve performance comparable with models on the COCO leaderboard from late 2016. This work uses the existing COCO 2017 Keypoint Detection dataset. The final model performs very well on most images, especially those containing well-separated people with the subject centered in frame. It struggles with images containing highly overlapped people or heavily occluded or articulated keypoints.

## Introduction

Our group created a stacked hourglass network that was trained on the Common Objects in Context (COCO) dataset. Our network predicts a maximum of 17 keypoints spanning the full human body on a 2D image. A number of challenges make HPE a difficult problem domain; these challenges include variability in human appearance and physique, environment lighting and weather, occlusions from other objects, self-occlusions from overlapping joints, complexity of movements of the human skeleton, and the inherent loss of information with a 2D image input. This largely unsolved problem enabled us to explore many novel and creative approaches, enriching our learning experience. We are excited to report the results we yielded.

## Method

### Problem Formulation

HPE systems can be categorized into 2D vs 3D and single-person vs multi-person. To improve the feasibility of our project, we have focused on single-frame single-person monocular RGB images. Current state-of-the-art techniques for 2D single-person HPE can be categorized into two categories: regression on absolute joint position, or detection on joint locations with heat maps. Since a direct mapping from the input space to joint coordinates is a highly non-linear problem, heat-map-based approaches have proven to be more robust by including small-region information. Thus, we have chosen the heat map approach.

There are three different types of models used with full body HPE: kinematic, contour, and volumetric, as shown in Fig. \ref{fig:body_model} \cite{Chen_2020}. A kinematic model resembles a stick-figure skeleton. The contour model consists of 2D squares and rectangles that represent the body, and the volumetric model represents the body with 3D cylinders. The kinematic model is the simplest model to perform loss metric computations, and thus is preferred by our group as a scope-limiting decision to simplify the problem space. Our goal is to predict a kinematic model for the individual in each picture.

We chose to use the COCO Keypoint dataset \cite{coco_data}. This dataset consists of 330 K images, of which 200 K are labelled. There are pre-sorted subsets of this dataset specific for HPE competitions: COCO16 and COCO17. These contain 147 K images labelled with bounding boxes, joint locations, and human body segmentation masks. We originally considered using DensePose, which is a highly detailed manually annotated subset of the COCO dataset, but found it does not offer joint coordinate labels. Another popular dataset is the MPII dataset, which consists of 41 K labelled images split into 29 K train and 12 K test. We originally planned to use this for validating our model’s performance.

### Dataset

TODO

### Network Architecture

The network is based on Newell 2016's stacked hourglass network. It consists of a series of stacked U-Nets. The network gets its name because the U-Nets resemble hourglass structures. Each U-Net, shown in Fig. \ref{fig:single_hourglass} is a lightweight encoder-decoder structure that consists of residual blocks with either convolution or upsampling applied for the encoder and decoder stages, respectively. Unlike typical U-Nets that have proven successful for problem domains such as semantic segmentation, this network does not use unpooling or deconvolutional layers during the decoder stage. Instead, nearest neighbour upsampling is used. Skip connections link feature levels of the same spatial resolution in the encoder and decoder stages. This structure allows the network to combine information from deep abstract features, and local high-resolution information.

The input to the entire model is a RGB image of resolution 256x256. Since performing operations at original resolution is expensive in compute and memory, all the internal hourglass stacks have a max resolution of  64x64. Thus, between the input layer and the first hourglass block, the input resolution is brought down from 256x256 to 64x64 by using the following operations: a 7x7 convolutional layer with stride 2, a residual module, and max pooling.

The input to each hourglass block is at a resolution of 64x64. The encoder block performs top-down processing, where the image spatial domain is decreased while increasing feature depth. After each convolution block, max pooling is applied to reduce in spatial size. The network also branches off at the pre-pooled resolution to apply more convolutions to form the \textit{skip connections}. At the smallest stage in the middle of the hourglass, which is denoted the \textit{bottleneck}, the network is at a resolution of 4x4 pixels. Here, the convolution operations can compare global features in the image. To reconstruct the original resolution, the network applies nearest-neighbour upsampling and joins feature information from the skip connections. The final resolution is identical to the input at 64x64, and the network is fully symmetric.

At the output resolution, the network predictions are constructed by applying two rounds of 1x1 convolutions to the extracted features. This forms a series of 17 one-channel heatmaps, one for each joint, where the intensity of the pixel value corresponds to the probability that a joint is found at that location. This intermediate prediction is added with the feature map and used as input to the next layer. In essence, each block in this architecture performs a refinement of predictions generated by the previous block. The number of hourglass blocks does not affect the output resolution, since each block is symmetric.

### Prediction

TODO

### Evaluation

TODO
OKS and PCK
