# Multi-modal gait recognition based on CNNs

Francisco M. Castro and Manuel J. Marin-Jimenez

This library contains support Matlab code for [1] and [2].
If you find useful this code, please, cite [1] or [2].

### Prerequisites
1. MatConvNet library: http://www.vlfeat.org/matconvnet/
2. MexConv3D (for 3D convs): https://github.com/pengsun/MexConv3D
3. Download the test data and models into their respective folders. The links are in the README file included in each folder

This code has been tested on Ubuntu 18.04 with Matlab 2017b.

### Quick start
Let's assume that you have placed _cnngaitmm_ library in folder `<cgdir>`. 
Start Matlab and type the following commands:

```
cd <cgdir>
startup_cnngait
demo_TUM
demo_CASIA
demo_TUM_multimodal
demo_CASIA_multimodal
```



