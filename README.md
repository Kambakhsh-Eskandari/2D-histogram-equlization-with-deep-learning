# 2D-histogram-equlization-with-deep-learning
A deep learning model to predict the cdf of an image

The architecture of the model is represented by the image down below, keep in mind that since the cdf of a grayscale image has 256 values thus we expect a model which gives us a vector of 256 values as output.

![183244900-e07a4efd-9ca5-4941-8d97-79ec505663a8](https://user-images.githubusercontent.com/83129774/211059017-5083e286-c495-400e-b7b3-f271205b232e.png)

This model is train on a costume dataset and we should keep in mind that tensorflow only accepts data as tensors thus our data should always be represented as (B, W, H, C). Also every image should be resized before performing feed-forward operation unless our model is a fully convolutional network.
