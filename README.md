# Blurring the Background of a Video with Neural Network

This repository contains code for blurring the background of a video using a neural network. The code is implemented in Python using the TensorFlow library.

## Usage

To run the code, you need to have Python and TensorFlow installed.


After installing TensorFlow, you can clone the repository and run the code.


## How it works

The code takes a video as input and uses a neural network to predict a binary mask that separates the foreground from the background. The predicted mask is then used to blur the background. The neural network is trained on a large dataset of images to learn how to distinguish between the foreground and background.

## Results

The code produces a blurred video as output, with the background being blurred and the foreground remaining in focus. The quality of the results depends on the quality of the neural network and the input video. The code is intended as a proof of concept and further improvement is possible by training the neural network on a larger and more diverse dataset.
