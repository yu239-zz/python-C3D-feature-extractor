# python-C3D-feature-extractor
A python script for setting up C3D feature extractor and post-processing features

This python script provides an easy use of the C3D feature extractor proposed in 

> D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, 
  "Learning Spatiotemporal Features with 3D Convolutional Networks", ICCV 2015 ([github](https://github.com/facebook/C3D))

(First you need to install C3D following Caffe installation instructions.)  
  
Given a list of video paths, the script
* converts each video to a sequence of frames (jpg) (note: directly feeding videos to C3D model might lose frames)
* automatically configures the prototxt, the input frame list, the output prefix list, and the bash script
* collects the outputs from C3D for each video, converting them from the original binary format to Python pickle objects

By default, the script only extracts fc6-1 features. You can modify it according to your needs.
