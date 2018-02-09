# Detect building in Aerial Image

## Pre-requesite 

Please install all the requirements:

```
pip3 install -r requirements.txt
```

We use an old version of Keras (Keras 1.2)

## Dataset

Create a file directory _raw/_ and _save/_ 

The raw directory will contain two directories:

1. raw/train:

Add your training data: 1.tif, 2.tif, 3.tif, etc.
And your training mask data: 1_mask.tif, 2_mask.tif, 3_mask.tif

All in the directory raw/train

2. raw/test:

Add your testing data: 1.tif, 2.tif, 3.tif 

## Getting started

1. Transforming the data:

```
> python data.py
```

This will collect, preprocess and store your training data and testing data in numpy array file (.npy)

2. Training the model and predicting:

```
> python train.py
```

This will train your model, save it as unet.hdf5 and use it to predict your testing data.

3. Generating .png output:

```
> python convert.py
```

This will convert the .npy prediction of the testing data into a .png file that you can find in the binary_image/ folder.

## Authors

* Akrem Bahri
* Yassine Belmamoun
