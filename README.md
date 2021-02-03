## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project is an image classification based on Bag of Visual Words.
	
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

If you want to train toyr classifier, you need to pass the directory path:
```
data_path = Path('train/') 
```
To run this project:

```
$ python main.py
```
