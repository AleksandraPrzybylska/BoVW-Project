## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is an image classification based on Bag of Visual Words.

How it works?
1. Extract local features from image using SIFT detector and descriptor.
2. Make clustering using algorithm MiniBatchKMeans with number of clusters equal 800.
3. Compare these features with visual words and create histograms for each image for the train and test dataset.
4. Predict the class of test images by comparing each histogram  of train images.
	
## Technologies
Project is created with:
* Python version: 3.8.6
* Numpy version: 1.19.4
* OpenCV version: 4.2.0.34
* Sklearn version: 0.22.2.post1
	
## Setup
All the necessary libraries are in the requirements file. You can install with a command:
```
$ pip install -r requirements.txt
```

If you want to train your classifier, you need to pass the directory path:
```python
data_path = Path('train/') 
```

To run this project:
```
$ python main.py
```
