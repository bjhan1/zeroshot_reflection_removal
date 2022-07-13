# Zero-Shot Learning for Reflection Removal of Single 360-Degree Image

This is the official repository for the reflection removal method published in ECCV2022.

## Background
- Supervised learning causes the burden of colloecting pairs of images to train the networks. 
- Furthermore, the existing reflection removal methods suffers from the domain gap between datasets. They are usually be able to remove only blurred reflection artifacts.
- 360-degree images include the reflected scenes which can be used to recognize and remove the intense reflection artifacts in the glass regions.


## Assumptions
- This work focuses on the image decomposition by adapting zero-shot learning. It requies the following assumptions.
  - A camera and a flat glass should be vertically standing on the ground.
  - The glass region is located in the center of  the 360-degree images.
  
## Preparation
### Initial weights
The pre-trained auto-encoder in arbirary datasets provides a good initial state.

Download the initial weights in [here](https://drive.google.com/file/d/1mM1WUlKTDwAC3a5TOpSnRgDFhb6YfFjK/view?usp=sharing)

### Datasets
We collected 30 real 360-degree images for reflection removal. You can download them in [here](https://drive.google.com/file/d/1woQJzJzvj0-FsRqOsGQTE1uDvBBZSyM0/view?usp=sharing).

For reflection removal, we believe the qualitative comparison in real images is quite more important than the quantitative comparison in synthetic images. If you need, you can download some synthetic 360 images used in the paper in [here]()

According to the default setting, you'd like to set your folder tree as follows.

```
zeroshot_reflection_removal
|-- data
|    |-- 360_reflection_images
|    |      |-- glass
|    |      |-- pano
|    |      |-- refer
|    |-- synthetic_reflection_images
|           |-- glass
|           |-- pano
|           |-- refer
|           |-- trans
|-- weights
|    |-- initial-epoch1.pth
|-- main.py
|-- main.sh
|
```

## Inference 
You can run the bash file with some arguments.
```python
bash main.sh
````
