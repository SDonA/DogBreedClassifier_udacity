### Table of Contents

1. [Project Overview](#Overview)
3. [Project Motivation](#motivation)
2. [Installation Instruction](#installation)
3. [File Descriptions](#files)
4. [Sample Prediction Results](#results)
5. [Licensing, Author, Acknowledgement, References](#licensing)


## Project Overview<a name="motivation"></a>

Computer vision is one of the major technological advances today - sitting at the the heart of processes/activities in various industries such as the automobile, medicine, security, communication, etc. In this project a Dog Breed classification system is developed using Convolutional Neural Networks.

## Project Motivation<a name="motivation"></a>

This is one of the projects recommended for the Udacity Capstone Project for the award of a Nanodegree in Data Science. With a flair for computer vision and deeplearning, I Was particularly drawn to take on this project as a dog lover, with an end game in mind of developing this further into a mobile and web app.

## Installation Instruction<a name="files"></a>

### Dependencies

- face_recognition==1.3.0
- ipython==8.0.1
- ipywidgets==7.6.3
- matplotlib==3.3.4
- numpy==1.19.2
- opencv_python==4.5.5.62
- Pillow==9.0.0
- torch==1.8.0
- torchvision==0.9.0
- tqdm==4.60.0

Face_recognition requires installation of dlib. Refer to this [link](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f) on how to install dlib and its dependencies
Info on Face recognition is located [here](https://face-recognition.readthedocs.io/en/latest/readme.html)

## File Descriptions <a name="files"></a>

The files neccessary to run the code are are below
1. dog_app_pytorch_SDA.ipnyb(file) - Main jupyter notebook file
2. haarcacades (folder) - contains OpenCV face recognition models
3. OptionalFaceDetector (folder) - contains OpenCV DNN model files
4. saved_models (folder) - contains optimal models saved after the various training steps in the notebooks
5. ./data/dog_images (folder)  - contains the training, validation and test dog image datasets
6. ./data/ifw (folder)  - contains human images
**NOTE**: Due to space limitations the data folder is hosted on my google drive [HERE](https://drive.google.com/drive/folders/1aHZcWOAUQGbjoQEgwh34FtjMQ3q89k65?usp=sharing)

**Steps to run the program:**

## Sample Prediction Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@succatt/key-parameters-impacting-mortality-in-heart-failure-hf-patients-a-data-driven-quest-2d9c273c46fc).




![plot](./images/Brittany_02625.jpg)

## Licensing, Author, Acknowledgement, References]<a name="licensing"></a>

Credits goes to Saurabh Shahane for the dataset provided on kaggle.
You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/saurabhshahane/in-hospital-mortality-prediction).  Feel free to use the code here as you would like! 

















## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*. on Anaconda (jupyter notebook)

Required Libraries are:

pandas
numpy
seaborn
scipy
sklearn
time
matplotlib
