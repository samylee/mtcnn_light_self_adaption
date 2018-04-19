# mtcnn_light_self_adaption
mtcnn-light version of adaptive image size

All the friends used to know that mtcnn-light needs to set up the width and height of the image in advance to initialize the network, which is not conducive to the face detection of different sizes of image input.

This project consummate the mtcnn-light version. We only need to initialize the network once, and we can detect faces for different sizes of images.

Opt: 

1.Adds a minimum face parameter to face detection.

2.The error of mtcnn-light in setting minimum size is solved.

3.Blog address: https://blog.csdn.net/samylee/
