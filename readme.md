# Video Backbone

### What's this?
Benchmark of video action classification for common CNN architectures. Implemented in PyTorch v1.3.1.

### Supported networks

| Network | #parameters (exclude final classifier) |
|----|----|
| 2d3d-ResNet18	| 31.82 M |
| 2d3d-ResNet34	| 60.80 M |
| 2d3d-ResNet50	| 44.74 M |
| 3d-ResNet18	| 33.15 M |
| 3d-ResNet34	| 63.46 M |
| 3d-ResNet50	| 46.14 M |
| I3D	| 12.29 M |
| S3D	| 7.910 M |
| S3D-G | 9.098 M |

### Files
* `backbone/` has all backbone models
* `model.py` gives an example of classifier with S3D backbone. 

### Notes
* benchmarks will come soon

### Link
* ResNet-2d3d is used in [SlowFast](https://github.com/facebookresearch/SlowFast),
[DPC](https://github.com/TengdaHan/DPC), etc.
* ResNet-3d is used in many papers, early ones like [Hara et al.](https://arxiv.org/abs/1711.09577)
* I3D from [Carreira and Zisserman](https://arxiv.org/abs/1705.07750)
* S3D/S3D-G from [Xie et al.](https://arxiv.org/abs/1712.04851)
