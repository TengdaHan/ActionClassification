# Video Backbone

### What's this?
Implementation of common video feature extractors with some benchmarks, in PyTorch v1.3.1.

### Content

* S3D from Xie et al. [Rethinking Spatiotemporal Feature Learning For Video Understanding](https://arxiv.org/abs/1712.04851)
* S3D-G comming soon

### Files
* `s3d.py` is the S3D feature extractor
* `model.py` gives an example of classifier using S3D backbone. 

### Notes
* more benchmarks will come later

### Reference
* S3D paper: [Rethinking Spatiotemporal Feature Learning For Video Understanding](https://arxiv.org/abs/1712.04851) 
* S3D/S3D-G tensorflow implementation: [here](https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py)
* S3D code is modified from [here](https://github.com/qijiezhao/s3d.pytorch) with tiny corrections
