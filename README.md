# Multi-modal gait recognition based on CNNs

Francisco M. Castro and Manuel J. Marin-Jimenez

This code run the test on the normal scenario for TUM-GAID and CASIA-B. For the other scenarios, you only have to download the datasets and build the corresponding imdbs.

The models included with the code are:
- CNN based on 3D convolutions using optical flow as input for TUM-GAID
- CNN based on 3D convolutions that performs the fusion of optical flow, gray and depth modalities for TUM-GAID.
- CNN based ResNet using gray as input for CASIA-B.
- CNN based on 3D convolutions that performs the fusion of optical flow and gray for CASIA-B.

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



