# Project-II

In this Project II, I perform unsupervised learning tumor recognition with BraTS2020 dataset
(https://www.med.upenn.edu/cbica/brats2020/data.html)
With the unsupervised learning method, in this problem I use the K - means algorithm to segment the image into K regions that support the identification of the container to be searched.

With the BraTS2020 dataset consisting of 369 data files in the training folder, each file includes 5 small data files, which are files in the format .nii.gz or .nii.To be able to read medical image files need to use nibabel library: 
# !pip install nibabel

Before starting to build a tumor segmentation program, it is necessary to check the data and classify the data. Because the problem uses uncontrolled machine learning, it is necessary to classify the test data into many groups for easy processing and feature extraction. The data is divided into 4 subgroups based on the intensity and nature of the image data.

Besides, we also need to visualize data to be able to see and understand what we are doing:

![image](https://user-images.githubusercontent.com/68015472/148993166-5072e49c-4230-414b-9a15-1ed9bd8f3efe.png)

