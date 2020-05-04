# Video Backbone

### What's this?
Benchmark of video action classification for common CNN architectures. Implemented in PyTorch v1.3.1.

### Supported architectures
* ResNet-2d3d (18, 34, 50, ...)
* ResNet-3d (18, 34, 50)
* I3D
* S3D, S3D-G

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
