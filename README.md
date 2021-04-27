## Dog-Breed-Image-Classification-Using-CNNs-And-Transfer-Learning-In-Pytorch
Udacity Deep Learning Nanodegree Project #2.

 * This is a repo for building a Dog Breed Identification App, a project using CNN as a part of Udacity Deep Learning NanoDegree.
 * It is implemented by using PyTorch library.
 * You can refer to Original Udacity repo [here](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/project-dog-classification)

## Project Synopsis

Welcome to the Convolutional Neural Network Project in Udacity Deep Learning NanoDegree Program.In this project,you will learn how to build a pipeline that can be used
within a web or mobile app to process real-world,user-supplied images.Given an image of dog,your algorithm will identify an estimate of the canine's breed.If provided an image of  human,the algorithm will identify the resembling Canine breed.

![image](https://github.com/udacity/deep-learning-v2-pytorch/raw/master/project-dog-classification/images/sample_dog_output.png) ![image](https://user-images.githubusercontent.com/54153994/115130002-e7cc7800-9feb-11eb-890e-d7aa82130972.png)

Exploring the CNN models,you will get an idea about how to make vital decisions about the user experience for your app.My goal is that by completing this project,you will understand the challenges involved in grouping together a series of models which are designed to perform various tasks in data preprocessing steps.Each model has its own advantanges and disadvantages,designing and engineering a real-world application often involves giving upfront solutions to the problem without finding a perfect answer.

## Project Information
* Intro
* Step 0: Import Datasets
   * Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
   * Download the [human_dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

* Step 1: Detect Humans Accuracy
* Step 2: Detect Dog Accuracy
* Step 3: Create a CNN to Classify Dog Breeds(from scratch)
* Step 4: Create a CNN to Classify Dog Breeds(using Transfer Learning)
* Step 5: Writing Own Algorithm
* Step 6: Testing Own Algorithm

## Topics Related To The Project
* Machine learning
* Deep Learning
* Classification
* Convolutional Neural Networks
* Regularization
* FeedForward Neural Networks
* Backward propagation
* Activation Functions
* Hyper-Parameters Tuning
* Pytorch

<h2>Programming Languages,Packages/Libraries and IDEs</h2>

[Python 2.7 or Higher](https://www.python.org/downloads/)

[Numpy](https://pypi.org/project/numpy/)

[Matplotlib](https://pypi.org/project/matplotlib/)

[Pandas](https://pypi.org/project/pandas/)

[Jupyter Notebook](https://jupyter.org/install)

[OpenCV](https://opencv.org/)

<h2>CNN Architecture Built From Scratch</h2>

#####
Net(
   (Conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
   (Conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
   (Conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
   (Conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
   (Conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  
   (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
   (fc1): Linear(in_features=12544, out_features=512, bias=True)
  
   (fc2): Linear(in_features=512, out_features=512, bias=True)
  
   (fc3): Linear(in_features=512, out_features=133, bias=True)
  
   (dropout): Dropout(p=0.5)
  )
  
  ​	Achieved an Accuracy up to **22%** with **30 epochs**
  
  ## Transfer Learnings

  Used a Pre-trained Model **VGG-16** for transfer learning

  ​ Achieved an Accuracy up to **78%** with **15 epochs**


   
