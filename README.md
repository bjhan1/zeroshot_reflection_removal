# Zero-Shot Learning for Reflection Removal of Single 360-Degree Image

This is the official repository the reflection removal method published in ECCV2022.

## Inroduction
The proposed method provides a zero-shot learning approach to avoid the burden of colloecting pairs of images to train the networks by supervised learning. It uses a single 360-degree image and supress or remove the reflection artifacts in the glass region.
Note that the glass localization is byeond our scope. 


## Initia model weights
Though the proposed method trains the network parameters in the test time for each 360-degree image, the pre-trained auto-encoder in arbirary dataset provides better performance.
Download the initial model weights in [here](https://drive.google.com/file/d/1woQJzJzvj0-FsRqOsGQTE1uDvBBZSyM0/view?usp=sharing)

## Dataset
All 360-degree images for reflection removal can be downloaded in [here]()


## Inference 
```python
bash main.sh
````
