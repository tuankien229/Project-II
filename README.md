# Project-II

In this Project II, I perform unsupervised learning tumor recognition with BraTS2020 dataset
(https://www.med.upenn.edu/cbica/brats2020/data.html)
With the unsupervised learning method, in this problem I use the K - means algorithm to segment the image into K regions that support the identification of the container to be searched.

With the BraTS2020 dataset consisting of 369 data files in the training folder, each file includes 5 small data files, which are files in the format .nii.gz or .nii.To be able to read medical image files need to use nibabel library: 
# !pip install nibabel

Before starting to build a tumor segmentation program, it is necessary to check the data and classify the data. Because the problem uses uncontrolled machine learning, it is necessary to classify the test data into many groups for easy processing and feature extraction. The data is divided into 4 subgroups based on the intensity and nature of the image data.

Besides, we also need to visualize data to be able to see and understand what we are doing:

![image](https://user-images.githubusercontent.com/68015472/148993166-5072e49c-4230-414b-9a15-1ed9bd8f3efe.png)

Applying the K - means segmentation algorithm combined with image superposition to remove the unsuitable regions, we obtain the desired area:

![image](https://user-images.githubusercontent.com/68015472/148994874-2c51527e-3589-48fd-b98f-a62e70ee79a9.png)

![image](https://user-images.githubusercontent.com/68015472/148994892-932a6aac-e057-4b81-a6a2-bd78e1105364.png)

![image](https://user-images.githubusercontent.com/68015472/148994904-8374f159-fc79-4579-9557-6e6dc371486e.png)

![image](https://user-images.githubusercontent.com/68015472/148994914-80b6cc6d-c086-41f0-b07b-96af0f4d178b.png)

![image](https://user-images.githubusercontent.com/68015472/148994935-49357712-3904-42ad-9d94-5f207c16a2d7.png)

![image](https://user-images.githubusercontent.com/68015472/148994966-3b8f60df-25ba-4e70-a0d4-3aa4fde36cdc.png)

To be able to calculate the exact recognition rate of this problem, I use the Dice function to compare and calculate the same ratio with the given container. 

![image](https://user-images.githubusercontent.com/68015472/148995476-a08cf789-863e-4edd-abb6-5eb5ac9bff9b.png)

![image](https://user-images.githubusercontent.com/68015472/148995438-360ccfa4-fc8e-4fc6-8854-615630528fa9.png)

# The result obtained after using the Dice function I got an accuracy rate of 81% accuracy rate.
